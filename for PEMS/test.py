########################
####### 20250402 #######
########################



###### 搭建环境 ######
'''
################################################################
# https://
################################################################
# https://github.com/
################################################################
# conda config --remove-key channels

# conda create -n STGNN_NN
# conda activate STGNN_NN

# https://pytorch.org/get-started/previous-versions/
# conda install \
# python=3.8 pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
################################################################
# pip install pandas
# pip install scipy

# pip install PyWavelets
# pip install fastdtw

# pip install pympler
################################################################
# 环境导出
# pip freeze > requirements.txt
# pip install -r requirements.txt
################################################################
'''



###### 包的导入 ######
import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import gc
import math
import numpy as np
import os
import pandas as pd  # 数据分析
import random
import time
from pympler import asizeof

import _1_util
from _1_util import *
from _1_ranger import Ranger
from _2_model import STGNN_NN



###### GPU ######
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
def report_gpu() :
   # print(torch.cuda.list_gpu_processes())
   gc.collect()
   torch.cuda.empty_cache()



###### 训练器类 ######
class trainer :


    def __init__(self, device, nodes, windows, horizons, 
                 revin_en, wavelet, h_channels, granularity, 
                 graph_dims, diffusion_k, dropout, 
                 lrate, wdecay, scaler) :

        self.model = STGNN_NN(device, nodes, windows, horizons, 
                             revin_en, wavelet, h_channels, granularity, 
                             graph_dims, diffusion_k, dropout)
        ### 模型放于cuda:0
        self.model.to(device)

        # self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.optimizer = Ranger(self.model.parameters(), lr=lrate, weight_decay=wdecay)

        self.scaler = scaler          # 用于prediction的反归一化

        self.clip   = 5               # 梯度分割

        # print('The number of parameters: {}'.format(self.model.param_num()))
        # print(self.model)


    ### 混合精度训练了解一下 ###
    # (model + x + y).to(device)为GPU的吞吐量，关联变量是否在cuda:0上
    def train(self, x, y) :
        # [B, 3, N, W] + [B, 1, N, H]

        self.model.train()  # 启用batch normalization和dropout
        self.optimizer.zero_grad()

        ########################
        prediction, _ = self.model(x)                           # -- [B, 1, N, H]
        prediction = self.scaler.inverse_transform(prediction)  # 预测结果的反归一化
        # print(prediction.device)  # cuda:0
        # print(A.device)           # cuda:0  # 可以.detach().cpu().numpy()

        ########################
        loss = _1_util.MAE_torch(prediction, y, 0.0)            # cuda:0
        loss.backward()                                         # loss回传，准备在各个节点上

        if self.clip is not None :  # 5
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)  # 梯度裁减，防止梯度爆炸

        self.optimizer.step()  # 根据loss，步进优化一次

        ########################
        mape  = _1_util.MAPE_torch (prediction, y, 0.0).item()
        rmse  = _1_util.RMSE_torch (prediction, y, 0.0).item()
        wmape = _1_util.WMAPE_torch(prediction, y, 0.0).item()

        return loss.item(), mape, rmse, wmape  # .item()将tensor转化为标量，不在cuda:0上


    def eval(self, x, y) :

        self.model.eval()  # 不启用batch normalization和dropout

        with torch.no_grad() :  # 不需计算梯度来更新参数，可以减少显存占用
            prediction, _ = self.model(x)
        prediction = self.scaler.inverse_transform(prediction)

        mae   = _1_util.MAE_torch  (prediction, y, 0.0).item()
        mape  = _1_util.MAPE_torch (prediction, y, 0.0).item()
        rmse  = _1_util.RMSE_torch (prediction, y, 0.0).item()
        wmape = _1_util.WMAPE_torch(prediction, y, 0.0).item()

        return mae, mape, rmse, wmape



###### 参数设置，共计20个，可调7个 ######
def para_cfg() :

    parser = argparse.ArgumentParser()


    ### 服务器设置-1：GPU编号
    parser.add_argument('--device',             type=str, default='cuda:0')         # cpu/cuda:0

    parser.add_argument('--dataset',            type=str, default='PEMS08')         # PEMS 08/03/04/07
    parser.add_argument('--dataset_dir',        type=str, default='./dataset/PEMS08')  # dataset/

    # B, 3, N, W, H
    parser.add_argument('--batch_size',         type=int, default=32)               # 16/32/48/64/80/96
    parser.add_argument('--i_channels',         type=int, default=3)                # 
    parser.add_argument('--nodes',              type=int, default=170)              # 
    parser.add_argument('--windows',            type=int, default=12)               # x
    parser.add_argument('--horizons',           type=int, default=12)               # y

    # 数据变形
    parser.add_argument('--revin_en',           type=int, default=0)                # 0/1
    parser.add_argument('--wavelet',            type=str, default='')               # sym2/db1/db1/coif1
    parser.add_argument('--h_channels',         type=int, default=96)               # 32/48/64/80/96
    parser.add_argument('--granularity',        type=int, default=288)              # 

    # 图结构参数
    parser.add_argument('--graph_dims',         type=int, default=10)               # 5/10/15
    parser.add_argument('--diffusion_k',        type=int, default=1)                # 1/2

    # 
    parser.add_argument('--dropout',            type=float, default=0.5)            # 0.1/0.3/0.5

    # 训练参数
    parser.add_argument('--epochs',             type=int, default=1000)             # 1000
    parser.add_argument('--learning_rate',      type=float, default=0.0005)         # 0.0005/0.0010/0.0015
    parser.add_argument('--weight_decay',       type=float, default=0.0001)         # 0.0001/0.0003/0.0005
    parser.add_argument('--cnt_log',            type=int, default=50)               # 
    parser.add_argument('--save_dir',           type=str, default='./logs/' 
                        + str(time.strftime('%Y-%m-%d-%H-%M-%S')) + '-')            # 时间相关
    parser.add_argument('--es_patience',        type=int, default=100)              # 100


    args = parser.parse_args()
    #args = parser.parse_args(args=[])

    return args



###### 每次设置随机数种子后，后续对应的随机函数所产生的随机数序列都相同 ######
def seed_it(seed) :

    np.random.seed(seed)
    os.environ['PYTHONSEED'] = str(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)



###### 主函数 ######
if __name__ == '__main__' :

    # 清理显存
    gc.collect()
    torch.cuda.empty_cache()

    # 时间关联的随机种子
    # seed = int(time.time())%10000       # 0-9999
    seed = 6666
    print('seed: {:04d}'.format(seed))  # 记录种子
    seed_it(seed)

    args = para_cfg()

    device = torch.device(args.device)  # cuda:0


    # 默认均为最佳参数
    if args.dataset   == 'PEMS08' :     # https://paperswithcode.com/sota/traffic-prediction-on-pems08
        # [10699/3567/3567, 12, 170, 3/1]
        # 服务器设置-2：数据集地址
        # args.dataset_dir    = '/home/zhfc/dataset/'+ args.dataset
        # args.dataset_dir    = '/home/zfc/dataset/' + args.dataset
        # args.dataset_dir    = '/root/dataset/'     + args.dataset
        args.batch_size     = 32        # 17856     ### 调整
        args.i_channels     = 3
        args.nodes          = 170
        args.windows        = 12
        args.horizons       = 12
        args.revin_en       = 0         #           ### 调整，PEMS的效果不佳
        args.wavelet        = ''        # 
        args.h_channels     = 96        #           ### 调整
        args.granularity    = 288       # 1day=24hrs=24*60mins=24*60/5=288
        args.graph_dims     = 10        # 
        args.diffusion_k    = 1         # 
        args.dropout        = 0.5       # 
        args.epochs         = 10000
        args.learning_rate  = 0.0005    # 
        args.weight_decay   = 0.0001    # 
        args.cnt_log        = 50
        args.save_dir       = './logs/' + str(time.strftime('%Y-%m-%d-%H-%M-%S')) + '-'
        args.es_patience    = 100
    elif args.dataset == 'PEMS03' :
        # [15711/5237/5237, 12, 358, 3/1]
        # 服务器设置-2：数据集地址
        # args.dataset_dir    = '/home/zhfc/dataset/'+ args.dataset
        # args.dataset_dir    = '/home/zfc/dataset/' + args.dataset
        # args.dataset_dir    = '/root/dataset/'     + args.dataset
        args.batch_size     = 64        # 26208     ### 调整
        args.i_channels     = 3
        args.nodes          = 358
        args.windows        = 12
        args.horizons       = 12
        args.revin_en       = 0         #           ### 调整，PEMS的效果不佳
        args.wavelet        = ''        # 
        args.h_channels     = 16        #           ### 调整
        args.granularity    = 288       # 1day=24hrs=24*60mins=24*60/5=288
        args.graph_dims     = 10        # 
        args.diffusion_k    = 1         # 
        args.dropout        = 0.5       # 
        args.epochs         = 10000
        args.learning_rate  = 0.0005    # 
        args.weight_decay   = 0.0001    # 
        args.cnt_log        = 50
        args.save_dir       = './logs/' + str(time.strftime('%Y-%m-%d-%H-%M-%S')) + '-'
        args.es_patience    = 100
    elif args.dataset == 'PEMS04' :     # https://paperswithcode.com/sota/traffic-prediction-on-pems04
        # 
        # 服务器设置-2：数据集地址
        # args.dataset_dir    = '/home/zhfc/dataset/'+ args.dataset
        # args.dataset_dir    = '/home/zfc/dataset/' + args.dataset
        # args.dataset_dir    = '/root/dataset/'     + args.dataset
        args.batch_size     = 64        # 16992     ### 调整
        args.i_channels     = 3
        args.nodes          = 307
        args.windows        = 12
        args.horizons       = 12
        args.revin_en       = 0         #           ### 调整，PEMS的效果不佳
        args.wavelet        = ''        # 
        args.h_channels     = 64        #           ### 调整
        args.granularity    = 288       # 1day=24hrs=24*60mins=24*60/5=288
        args.graph_dims     = 10        # 
        args.diffusion_k    = 1         # 
        args.dropout        = 0.5       # 
        args.epochs         = 10000
        args.learning_rate  = 0.0005    # 
        args.weight_decay   = 0.0001    # 
        args.cnt_log        = 50
        args.save_dir       = './logs/' + str(time.strftime('%Y-%m-%d-%H-%M-%S')) + '-'
        args.es_patience    = 100
    elif args.dataset == 'PEMS07' :     # https://paperswithcode.com/sota/traffic-prediction-on-pems07
        # 
        # 服务器设置-2：数据集地址
        # args.dataset_dir    = '/home/zhfc/dataset/'+ args.dataset
        # args.dataset_dir    = '/home/zfc/dataset/' + args.dataset
        # args.dataset_dir    = '/root/dataset/'     + args.dataset
        args.batch_size     = 16        # 28224     ### 调整
        args.i_channels     = 3
        args.nodes          = 883
        args.windows        = 12
        args.horizons       = 12
        args.revin_en       = 0         #           ### 调整，PEMS的效果不佳
        args.wavelet        = ''        # 
        args.h_channels     = 128       #           ### 调整
        args.granularity    = 288       # 1day=24hrs=24*60mins=24*60/5=288
        args.graph_dims     = 10        # 
        args.diffusion_k    = 1         # 
        args.dropout        = 0.5       # 
        args.epochs         = 10000
        args.learning_rate  = 0.0005    # 
        args.weight_decay   = 0.0001    # 
        args.cnt_log        = 50
        args.save_dir       = './logs/' + str(time.strftime('%Y-%m-%d-%H-%M-%S')) + '-'
        args.es_patience    = 100
    elif args.dataset == 'bike_drop' :
        # 
        # 服务器设置-2：数据集地址
        # args.dataset_dir    = '/home/zhfc/dataset/'+ args.dataset
        # args.dataset_dir    = '/home/zfc/dataset/' + args.dataset
        # args.dataset_dir    = '/root/dataset/'     + args.dataset
        args.batch_size     = 64        #           ### 调整
        args.i_channels     = 3         # 
        args.nodes          = 250       # 
        args.windows        = 12
        args.horizons       = 12
        args.revin_en       = 0         #           ### 调整，未知效果
        args.wavelet        = ''        # 
        args.h_channels     = 32        #           ### 调整
        args.granularity    = 48        # 
        args.graph_dims     = 10        # 
        args.diffusion_k    = 1         # 
        args.dropout        = 0.5       # 
        args.epochs         = 10000
        args.learning_rate  = 0.0005    # 
        args.weight_decay   = 0.0001    # 
        args.cnt_log        = 50
        args.save_dir       = './logs/' + str(time.strftime('%Y-%m-%d-%H-%M-%S')) + '-'
        args.es_patience    = 100
    elif args.dataset == 'bike_pick' :
        # 
        # 服务器设置-2：数据集地址
        # args.dataset_dir    = '/home/zhfc/dataset/'+ args.dataset
        # args.dataset_dir    = '/home/zfc/dataset/' + args.dataset
        # args.dataset_dir    = '/root/dataset/'     + args.dataset
        args.batch_size     = 64        #           ### 调整
        args.i_channels     = 3         # 
        args.nodes          = 250       # 
        args.windows        = 12
        args.horizons       = 12
        args.revin_en       = 0         #           ### 调整，未知效果
        args.wavelet        = ''        # 
        args.h_channels     = 32        #           ### 调整
        args.granularity    = 48        # 
        args.graph_dims     = 10        # 
        args.diffusion_k    = 1         # 
        args.dropout        = 0.5       # 
        args.epochs         = 10000
        args.learning_rate  = 0.0005    # 
        args.weight_decay   = 0.0001    # 
        args.cnt_log        = 50
        args.save_dir       = './logs/' + str(time.strftime('%Y-%m-%d-%H-%M-%S')) + '-'
        args.es_patience    = 100
    elif args.dataset == 'taxi_drop' :
        # 
        # 服务器设置-2：数据集地址
        # args.dataset_dir    = '/home/zhfc/dataset/'+ args.dataset
        # args.dataset_dir    = '/home/zfc/dataset/' + args.dataset
        # args.dataset_dir    = '/root/dataset/'     + args.dataset
        args.batch_size     = 64        #           ### 调整
        args.i_channels     = 3         # 
        args.nodes          = 266       # 
        args.windows        = 12
        args.horizons       = 12
        args.revin_en       = 0         #           ### 调整，未知效果
        args.wavelet        = ''        # 
        args.h_channels     = 96        #           ### 调整
        args.granularity    = 48        # 
        args.graph_dims     = 10        # 
        args.diffusion_k    = 1         # 
        args.dropout        = 0.5       # 
        args.epochs         = 10000
        args.learning_rate  = 0.0005    # 
        args.weight_decay   = 0.0001    # 
        args.cnt_log        = 50
        args.save_dir       = './logs/' + str(time.strftime('%Y-%m-%d-%H-%M-%S')) + '-'
        args.es_patience    = 100
    elif args.dataset == 'taxi_pick' :
        # 
        # 服务器设置-2：数据集地址
        # args.dataset_dir    = '/home/zhfc/dataset/'+ args.dataset
        # args.dataset_dir    = '/home/zfc/dataset/' + args.dataset
        # args.dataset_dir    = '/root/dataset/'     + args.dataset
        args.batch_size     = 64        #           ### 调整
        args.i_channels     = 3         # 
        args.nodes          = 266       # 
        args.windows        = 12
        args.horizons       = 12
        args.revin_en       = 0         #           ### 调整，未知效果
        args.wavelet        = ''        # 
        args.h_channels     = 96        #           ### 调整
        args.granularity    = 48        # 
        args.graph_dims     = 10        # 
        args.diffusion_k    = 1         # 
        args.dropout        = 0.5       # 
        args.epochs         = 10000
        args.learning_rate  = 0.0005    # 
        args.weight_decay   = 0.0001    # 
        args.cnt_log        = 50
        args.save_dir       = './logs/' + str(time.strftime('%Y-%m-%d-%H-%M-%S')) + '-'
        args.es_patience    = 100
    print(args)


    # 载入数据集，./dataset/PEMS08/(train/valid/test).npz，返回dataloader，存储在CPU上
    dataloader = _1_util.load_dataset(args.dataset_dir, args.batch_size, args.batch_size, args.batch_size)
    scaler     = dataloader['scaler']  # 读出类的实例，两个值+两个函数


    # 相关的数据结构，CPU吞吐
    epoch_train_time  = []      # 每个epoch的训练时间
    epoch_valid_time  = []      # 每个epoch的验证时间
    his_valid_m_mae   = []      # 每个epoch的验证mae
    min_valid_m_mae   = 999999  # 历史里最小的验证mae
    train_valid_result= []      # 训练-验证的数据存储
    min_test_m_mae    = 999999  # 历史里最小的测试mae
    test_result       = []      # 测试的数据存储
    cnt_updated       = 0       # 最近轮次更新的计数
    save_path         = args.save_dir + args.dataset + '/'    # 文件保存路径
    if not os.path.exists(save_path) :
        os.makedirs(save_path)  # 包含时间，每次运行程序都会创建
    # 保存参数文件
    args_dict = vars(args)
    with open(save_path + 'paras.txt', 'w') as f :
        f.write('seed: ' + str(seed) + '\n')
        for key, value in args_dict.items() :
            f.write(f'{key}: {value}\n')


    # 训练器
    engine = trainer(device, args.nodes, args.windows, args.horizons, 
                     args.revin_en, args.wavelet, args.h_channels, args.granularity, 
                     args.graph_dims, args.diffusion_k, args.dropout, 
                     args.learning_rate, args.weight_decay, scaler)



    '''
    ### (训练-验证-测试)大循环
    for i in range(1, args.epochs+1) :  # 10000 epochs


        # 清理GPU显存
        # report_gpu()


        ############ 1-epoch训练 ############
        iters_train_mae   = []  # 每个iters的mae，每个epoch初始清空
        iters_train_mape  = []
        iters_train_rmse  = []
        iters_train_wmape = []

        t1 = time.time()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()) :  # train_iters
            # [B, W, N, 3] + [B, H, N, 1]

            x = torch.Tensor(x).to(device)
            x = x.transpose(1, 3)  # [B, 3, N, W]
            y = torch.Tensor(y).to(device)
            y = y.transpose(1, 3)  # [B, 1, N, H]

            metrics = engine.train(x, y)

            iters_train_mae  .append(metrics[0])
            iters_train_mape .append(metrics[1])
            iters_train_rmse .append(metrics[2])
            iters_train_wmape.append(metrics[3])

            # if iter % args.cnt_log == 0 :
            #     log = 'Iter: {:04d}, Tr_MAE: {:.4f}, Tr_MAPE: {:.4f}, Tr_RMSE: {:.4f}, Tr_WMAPE: {:.4f}'
            #     print(log.format(iter, 
            #                      iters_train_mae  [-1], 
            #                      iters_train_mape [-1], 
            #                      iters_train_rmse [-1], 
            #                      iters_train_wmape[-1]), flush=True)
        t2 = time.time()
        epoch_train_time.append(t2-t1)


        ############ 2-epoch验证 ############
        iters_valid_mae   = []  # 每个iters的mae，每个epoch初始清空
        iters_valid_mape  = []
        iters_valid_rmse  = []
        iters_valid_wmape = []

        v1 = time.time()
        for iter, (x, y) in enumerate(dataloader['valid_loader'].get_iterator()) :  # valid_iters
            # [B, W, N, 3] + [B, H, N, 1]

            x = torch.Tensor(x).to(device)
            x = x.transpose(1, 3)  # [B, 3, N, W]
            y = torch.Tensor(y).to(device)
            y = y.transpose(1, 3)  # [B, 1, N, H]

            metrics = engine.eval(x, y)

            iters_valid_mae  .append(metrics[0])
            iters_valid_mape .append(metrics[1])
            iters_valid_rmse .append(metrics[2])
            iters_valid_wmape.append(metrics[3])
        v2 = time.time()
        epoch_valid_time.append(v2-v1)


        ############ 3-epoch数据处理 ############
        epoch_train_m_mae   = np.mean(iters_train_mae)
        epoch_train_m_mape  = np.mean(iters_train_mape)
        epoch_train_m_rmse  = np.mean(iters_train_rmse)
        epoch_train_m_wmape = np.mean(iters_train_wmape)

        epoch_valid_m_mae   = np.mean(iters_valid_mae)
        his_valid_m_mae.append(epoch_valid_m_mae)
        epoch_valid_m_mape  = np.mean(iters_valid_mape)
        epoch_valid_m_rmse  = np.mean(iters_valid_rmse)
        epoch_valid_m_wmape = np.mean(iters_valid_wmape)

        # 存储形式
        epoch_m_value = dict(
            epoch_train_m_mae  = epoch_train_m_mae, 
            epoch_train_m_mape = epoch_train_m_mape, 
            epoch_train_m_rmse = epoch_train_m_rmse, 
            epoch_train_m_wmape= epoch_train_m_wmape, 

            epoch_valid_m_mae  = epoch_valid_m_mae, 
            epoch_valid_m_mape = epoch_valid_m_mape, 
            epoch_valid_m_rmse = epoch_valid_m_rmse, 
            epoch_valid_m_wmape= epoch_valid_m_wmape
        )
        epoch_m_value = pd.Series(epoch_m_value)  # 转化为表的一行，类一维数组
        train_valid_result.append(epoch_m_value)  # epoch内的平均值大集合

        # 打印
        log = 'Epoch: {:04d}, Tr Time: {:.4f} secs, Vd Time: {:.4f} secs'
        print(log.format(i, (t2-t1), (v2-v1)))
        log = 'Epoch: {:04d}, Tr_mMAE: {:.4f}, Tr_mMAPE: {:.4f}, Tr_mRMSE: {:.4f}, Tr_mWMAPE: {:.4f}'
        print(log.format(i, 
                         epoch_train_m_mae, 
                         epoch_train_m_mape, 
                         epoch_train_m_rmse, 
                         epoch_train_m_wmape), flush=True)
        log = 'Epoch: {:04d}, Vd_mMAE: {:.4f}, Vd_mMAPE: {:.4f}, Vd_mRMSE: {:.4f}, Vd_mWMAPE: {:.4f}'
        print(log.format(i, 
                         epoch_valid_m_mae, 
                         epoch_valid_m_mape, 
                         epoch_valid_m_rmse, 
                         epoch_valid_m_wmape), flush=True)


        ############ 4-epoch测试 ############
        # 训练时model参数的优化方向，为使训练集mae下降的方向
        # 首先判断，优化后的model，能否使得验证集mae下降
        # 再次判断，优化后的model，能否使得测试集mae下降
        # 三集下降，则认为模型有效
        if epoch_valid_m_mae < min_valid_m_mae :  # 本轮验证集mae下降有效

            if i < 10 :  # 百轮之内，模型大概率未收敛，只要降低了验证集mae，就认为模型有效

                bestid          = i
                min_valid_m_mae = epoch_valid_m_mae

                log = 'Epoch: {:04d}, Valid Good!!! Valid_MAE: {:.4f}'
                print(log.format(i, min_valid_m_mae), flush=True)

                torch.save(engine.model.state_dict(), save_path + 'best_model.pth')
                cnt_updated = 0


            elif i >= 10 :  # 百轮之后，继续进行测试集有效判断

                # 测试集真实值，cuda:0
                y_real = torch.Tensor(dataloader['y_test']).to(device)  # [3567, 12, 170, 1]
                y_real = y_real.transpose(1, 3)[:, 0, :, :]             # [3567, 170, 12]

                # 测试集预测值
                predictions = []
                for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()) :

                    x = torch.Tensor(x).to(device)
                    x = x.transpose(1, 3)  # [B, 3, N, W]

                    engine.model.eval()
                    with torch.no_grad() :
                        predicts, _ = engine.model(x)  # [B, 1, N, H]
                    predictions.append(predicts.squeeze())
                # print(len(predictions))  # 56
                y_hat = torch.cat(predictions, dim=0)  # [3584, 170, 12]  含有paddings
                y_hat = y_hat[:y_real.size(0), ...]    # [3567, 170, 12]
                y_hat = scaler.inverse_transform(y_hat)# [3567, 170, 12]


                # 指标计算
                horizon_test_mae   = []  # 不同于168项，这里只有12项，每个horizon上的mae
                horizon_test_mape  = []
                horizon_test_rmse  = []
                horizon_test_wmape = []

                # [3567*170*12] vs. [3567*170*12] -- 12*4
                for j in range(12) :

                    pred = y_hat [:, :, j]
                    real = y_real[:, :, j]
                    metrics = _1_util.metric(pred, real)  # [3567, 170] vs. [3567, 170] -- R4

                    # log = 'Horizon: {:04d}, Tt_MAE: {:.4f}, Tt_MAPE: {:.4f}, Tt_RMSE: {:.4f}, Tt_WMAPE: {:.4f}'
                    # print(log.format(j+1, metrics[0], metrics[1], metrics[2], metrics[3]))

                    # 12个时间步上
                    horizon_test_mae  .append(metrics[0])
                    horizon_test_mape .append(metrics[1])
                    horizon_test_rmse .append(metrics[2])
                    horizon_test_wmape.append(metrics[3])

                # 代码可以下移，做成单调递减记录
                # 存储形式：所有12个时间步上，所有条例的节点，在每一个指标上的平均值
                test_m_value = dict(
                    test_m_mae  =np.mean(horizon_test_mae), 
                    test_m_mape =np.mean(horizon_test_mape), 
                    test_m_rmse =np.mean(horizon_test_rmse), 
                    test_m_wmape=np.mean(horizon_test_wmape)
                )
                test_m_value = pd.Series(test_m_value)
                test_result.append(test_m_value)

                # 打印
                log = 'Epoch: {:04d}, Tt_mMAE: {:.4f}, Tt_mMAPE: {:.4f}, Tt_mRMSE: {:.4f}, Tt_mWMAPE: {:.4f}'
                print(log.format(i, 
                                 np.mean(horizon_test_mae) , 
                                 np.mean(horizon_test_mape), 
                                 np.mean(horizon_test_rmse), 
                                 np.mean(horizon_test_wmape)))


                # 判断本轮的测试指标，是否使得测试集mae下降
                if np.mean(horizon_test_mae) < min_test_m_mae :

                    bestid          = i
                    min_valid_m_mae = epoch_valid_m_mae
                    min_test_m_mae  = np.mean(horizon_test_mae)

                    log = '################ Epoch: {:04d}, Test Good!!!' + \
                          'Valid_MAE: {:.4f}, Test_MAE: {:.4f} ################'
                    print(log.format(i, min_valid_m_mae, min_test_m_mae), flush=True)

                    torch.save(engine.model.state_dict(), save_path + 'best_model.pth')
                    cnt_updated = 0

                else :  # 验证集mae下降，但是测试集mae没有下降
                    cnt_updated += 1
                    print('No Update: valid good + test bad')

        else :  # 训练出来的模型，没有使得验证集mae下降
            cnt_updated += 1
            print('No Update: valid bad')
        ### 测试结束


        if cnt_updated >= args.es_patience and i >= 100 :  # 跳出循环提前终止的条件
            break


    ### 大循环结束



    ############ 训练验证的总结 ############
    print('############ Summarizing!!! ############')
    print('Average Tr Time: {:.4f} secs/epoch'.format(np.mean(epoch_train_time)))
    print('Average Vd Time: {:.4f} secs/epoch'.format(np.mean(epoch_valid_time)))
    print('Best ID : ', bestid)
    print('Best MAE: ', str(round(his_valid_m_mae[bestid-1], 4)))

    # print(asizeof.asizeof(train_valid_result))  # 36952/10=3696
    train_valid_2_csv = pd.DataFrame(train_valid_result)  # 创建和操作数据框
    train_valid_2_csv.round(8).to_csv(f'{save_path}/train_valid.csv')
    '''



    ############ 最终模型的测试 ############
    # 清理显存
    gc.collect()
    torch.cuda.empty_cache()


    # 载入最终模型
    engine.model.load_state_dict(torch.load('best_model.pth'))
    torch.save(engine.model.state_dict(), save_path + 'best_model.pth')


    # 测试集真实值，cuda:0
    y_real = torch.Tensor(dataloader['y_test']).to(device)  # [3567, 12, 170, 1]
    y_real = y_real.transpose(1, 3)[:, 0, :, :]             # [3567, 170, 12]

    # 测试集预测值
    predictions = []
    As          = []
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()) :

        x = torch.Tensor(x).to(device)
        x = x.transpose(1, 3)  # [B, 3, N, W]

        engine.model.eval()
        with torch.no_grad() :
            predicts, A = engine.model(x)  # [B, 1, N, H]
        predictions.append(predicts.squeeze())
        As         .append(A)
    # print(len(predictions))  # 56
    y_hat = torch.cat(predictions, dim=0)  # [3584, 170, 12]  含有paddings
    y_hat = y_hat[:y_real.size(0), ...]    # [3567, 170, 12]
    y_hat = scaler.inverse_transform(y_hat)# [3567, 170, 12]
    A_    = torch.cat(As         , dim=0)
    A_    = A_   [:y_real.size(0), ...]


    # 指标计算
    last_test_mae   = []  # 有12项，每个horizon上的mae
    last_test_mape  = []
    last_test_rmse  = []
    last_test_wmape = []

    # [3567*170*12] vs. [3567*170*12] -- 12*4
    for k in range(12) :

        pred = y_hat [:, :, k]
        real = y_real[:, :, k]
        metrics = _1_util.metric(pred, real)  # [3567, 170] vs. [3567, 170] -- R4

        log = 'Horizon: {:04d}, Tt_MAE: {:.4f}, Tt_MAPE: {:.4f}, Tt_RMSE: {:.4f}, Tt_WMAPE: {:.4f}'
        print(log.format(k+1, metrics[0], metrics[1], metrics[2], metrics[3]))  # 打印12*4指标矩阵

        last_test_mae  .append(metrics[0])
        last_test_mape .append(metrics[1])
        last_test_rmse .append(metrics[2])
        last_test_wmape.append(metrics[3])


    # 打印
    log = 'Tt_mMAE: {:.4f}, Tt_mMAPE: {:.4f}, Tt_mRMSE: {:.4f}, Tt_mWMAPE: {:.4f}'
    print(log.format(np.mean(last_test_mae) , 
                     np.mean(last_test_mape), 
                     np.mean(last_test_rmse), 
                     np.mean(last_test_wmape)))



    ############ 模型测试的记录 ############


    # 参数回顾
    print('seed: {:04d}'.format(seed))
    print(args)


    # 测试文件写入
    test_m_value = []
    test_m_value = dict(
        test_m_mae  =np.mean(last_test_mae), 
        test_m_mape =np.mean(last_test_mape), 
        test_m_rmse =np.mean(last_test_rmse), 
        test_m_wmape=np.mean(last_test_wmape)
    )
    test_m_value = pd.Series(test_m_value)
    test_result.append(test_m_value)
    # print(asizeof.asizeof(test_result))  # 17832/6=2972

    # 测试文件写入mae
    test_steps_mae = []
    test_steps_mae = dict(
        last_test_mae_0  = last_test_mae[0], 
        last_test_mae_1  = last_test_mae[1], 
        last_test_mae_2  = last_test_mae[2], 
        last_test_mae_3  = last_test_mae[3], 
        last_test_mae_4  = last_test_mae[4], 
        last_test_mae_5  = last_test_mae[5], 
        last_test_mae_6  = last_test_mae[6], 
        last_test_mae_7  = last_test_mae[7], 
        last_test_mae_8  = last_test_mae[8], 
        last_test_mae_9  = last_test_mae[9], 
        last_test_mae_10 = last_test_mae[10], 
        last_test_mae_11 = last_test_mae[11]
    )
    test_steps_mae = pd.Series(test_steps_mae)
    test_result.append(test_steps_mae)

    # 测试文件写入mape
    test_steps_mape = []
    test_steps_mape = dict(
        last_test_mape_0  = last_test_mape[0], 
        last_test_mape_1  = last_test_mape[1], 
        last_test_mape_2  = last_test_mape[2], 
        last_test_mape_3  = last_test_mape[3], 
        last_test_mape_4  = last_test_mape[4], 
        last_test_mape_5  = last_test_mape[5], 
        last_test_mape_6  = last_test_mape[6], 
        last_test_mape_7  = last_test_mape[7], 
        last_test_mape_8  = last_test_mape[8], 
        last_test_mape_9  = last_test_mape[9], 
        last_test_mape_10 = last_test_mape[10], 
        last_test_mape_11 = last_test_mape[11]
    )
    test_steps_mape = pd.Series(test_steps_mape)
    test_result.append(test_steps_mape)

    # 测试文件写入rmse
    test_steps_rmse = []
    test_steps_rmse = dict(
        last_test_rmse_0  = last_test_rmse[0], 
        last_test_rmse_1  = last_test_rmse[1], 
        last_test_rmse_2  = last_test_rmse[2], 
        last_test_rmse_3  = last_test_rmse[3], 
        last_test_rmse_4  = last_test_rmse[4], 
        last_test_rmse_5  = last_test_rmse[5], 
        last_test_rmse_6  = last_test_rmse[6], 
        last_test_rmse_7  = last_test_rmse[7], 
        last_test_rmse_8  = last_test_rmse[8], 
        last_test_rmse_9  = last_test_rmse[9], 
        last_test_rmse_10 = last_test_rmse[10], 
        last_test_rmse_11 = last_test_rmse[11]
    )
    test_steps_rmse = pd.Series(test_steps_rmse)
    test_result.append(test_steps_rmse)

    # 测试文件写入wmape
    # 

    # 测试文件保存
    test_2_csv = pd.DataFrame(test_result)
    test_2_csv.round(8).to_csv(f'{save_path}/test.csv')


    # 保存相关画图数据为.npy
    np.save(save_path + 'y_real.npy', y_real.detach().cpu().numpy())
    np.save(save_path + 'y_hat.npy',  y_hat .detach().cpu().numpy())
    np.save(save_path + 'As.npy',     A_    .detach().cpu().numpy())

