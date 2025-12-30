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
                 lrate, wdecay, scale) :

        self.model = STGNN_NN(device, nodes, windows, horizons, 
                             revin_en, wavelet, h_channels, granularity, 
                             graph_dims, diffusion_k, dropout)
        ### 模型放于cuda:0
        self.model.to(device)

        ### 损失函数放于了cuda:0
        self.loss_L1  = nn.L1Loss (size_average=False).to(device)
        self.loss_mse = nn.MSELoss(size_average=False).to(device)

        # self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.optimizer = Ranger(self.model.parameters(), lr=lrate, weight_decay=wdecay)

        self.scale  = scale           # 用于反归一化

        self.clip   = 5               # 梯度分割

        # print('The number of parameters: {}'.format(self.model.param_num()))
        # print(self.model)


    ### 混合精度训练了解一下 ###
    # (model + x + y).to(device)为GPU的吞吐量，关联变量是否在cuda:0上
    def train(self, x, y) :
        # [B, 3, N, W] + [B, N]

        self.model.train()  # 启用batch normalization和dropout
        self.optimizer.zero_grad()

        preds, _ = self.model(x)                      # [B, 3, N, W] -- [B, 1, N, H]
        preds = preds[:, :, :, -1].squeeze()          # [B, 1, N, H] -- [B, 1, N] -- [B, N]

        scale = self.scale.expand(y.size(0), -1)      # [B, N]
        preds = preds * scale                         # Hadamard，反归一化
        y     = y     * scale                         # Hadamard，反归一化

        loss = self.loss_L1(preds, y)                 # loss的种类
        loss.backward()                               # loss回传，准备在各个节点上

        if self.clip is not None :  # 5
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)  # 梯度裁减，防止梯度爆炸

        self.optimizer.step()  # 根据loss，步进优化一次

        return loss.item()


    def eval(self, x, y) :

        self.model.eval()  # 不启用batch normalization和dropout

        with torch.no_grad() :  # 不需计算梯度来更新参数，可以减少显存占用
            preds, _ = self.model(x)
        preds = preds[:, :, :, -1].squeeze()

        scale = self.scale.expand(y.size(0), -1)
        preds = preds * scale
        y     = y     * scale

        loss_L1  = self.loss_L1 (preds, y)
        loss_mse = self.loss_mse(preds, y)

        return loss_L1.item(), loss_mse.item(), y.size(0)*y.size(1), preds
        # R + R + B*N + [B, N]



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
    parser.add_argument('--wavelet',            type=str, default='')               # ///
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
    if args.dataset == 'electricity' :  # 
        # 
        # 服务器设置-2：数据集地址
        # args.dataset_dir    = '/home/zhfc/dataset/'+ args.dataset
        # args.dataset_dir    = '/home/zfc/dataset/' + args.dataset
        # args.dataset_dir    = '/root/dataset/'     + args.dataset
        args.batch_size     = 8         # 26304     ### 调整
        args.i_channels     = 1         # 
        args.nodes          = 321       # 321
        args.windows        = 168       # 168
        args.horizons       = 3         # 3/6/12/24 ### 调整
        args.revin_en       = 1         #           ### 调整
        args.wavelet        = ''        # 
        args.h_channels     = 16        #           ### 调整
        args.granularity    = 24        # 1day=24hrs=24/1=24
        args.graph_dims     = 10        # 
        args.diffusion_k    = 1         # 
        args.dropout        = 0.5       # 
        args.epochs         = 10000
        args.learning_rate  = 0.0005    # 
        args.weight_decay   = 0.0001    # 
        args.cnt_log        = 50
        args.save_dir       = './logs/' + str(time.strftime('%Y-%m-%d-%H-%M-%S')) + '-'
        args.es_patience    = 100
    elif args.dataset == 'exchange_rate' :  # 
        # 
        # 服务器设置-2：数据集地址
        # args.dataset_dir    = '/home/zhfc/dataset/'+ args.dataset
        # args.dataset_dir    = '/home/zfc/dataset/' + args.dataset
        # args.dataset_dir    = '/root/dataset/'     + args.dataset
        args.batch_size     = 8         # 7588      ### 调整
        args.i_channels     = 1         # 
        args.nodes          = 8         # 8
        args.windows        = 168       # 168
        args.horizons       = 3         # 3/6/12/24 ### 调整
        args.revin_en       = 1         #           ### 调整
        args.wavelet        = ''        # 
        args.h_channels     = 16        #           ### 调整
        args.granularity    = 1         # 1day=1/1=1
        args.graph_dims     = 10        # 
        args.diffusion_k    = 1         # 
        args.dropout        = 0.5       # 
        args.epochs         = 10000
        args.learning_rate  = 0.0005    # 
        args.weight_decay   = 0.0001    # 
        args.cnt_log        = 50
        args.save_dir       = './logs/' + str(time.strftime('%Y-%m-%d-%H-%M-%S')) + '-'
        args.es_patience    = 200
    elif args.dataset == 'solar_AL' :  # 
        # 
        # 服务器设置-2：数据集地址
        # args.dataset_dir    = '/home/zhfc/dataset/'+ args.dataset
        # args.dataset_dir    = '/home/zfc/dataset/' + args.dataset
        # args.dataset_dir    = '/root/dataset/'     + args.dataset
        args.batch_size     = 8         # 52560     ### 调整
        args.i_channels     = 1         # 
        args.nodes          = 137       # 137
        args.windows        = 168       # 168
        args.horizons       = 3         # 3/6/12/24 ### 调整
        args.revin_en       = 1         #           ### 调整
        args.wavelet        = ''        # 
        args.h_channels     = 16        #           ### 调整
        args.granularity    = 144       # 1day=24hrs=24*60mins=24*60/10=144
        args.graph_dims     = 10        # 
        args.diffusion_k    = 1         # 
        args.dropout        = 0.5       # 
        args.epochs         = 10000
        args.learning_rate  = 0.0005    # 
        args.weight_decay   = 0.0001    # 
        args.cnt_log        = 50
        args.save_dir       = './logs/' + str(time.strftime('%Y-%m-%d-%H-%M-%S')) + '-'
        args.es_patience    = 100
    elif args.dataset == 'traffic' :  # 
        # 
        # 服务器设置-2：数据集地址
        # args.dataset_dir    = '/home/zhfc/dataset/'+ args.dataset
        # args.dataset_dir    = '/home/zfc/dataset/' + args.dataset
        # args.dataset_dir    = '/root/dataset/'     + args.dataset
        args.batch_size     = 8         # 17544     ### 调整
        args.i_channels     = 1         # 
        args.nodes          = 862       # 862
        args.windows        = 168       # 168
        args.horizons       = 3         # 3/6/12/24 ### 调整
        args.revin_en       = 1         #           ### 调整
        args.wavelet        = ''        # 
        args.h_channels     = 16        #           ### 调整
        args.granularity    = 24        # 1day=24hrs=24/1=24
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


    # .txt ---- set_data: [L, W, N, 3] + [L, N]
    set_data = _1_util.load_dataset(
        device, args.dataset_dir, 0.6, 0.2, args.windows, args.horizons, args.granularity)
    # print(asizeof.asizeof(set_data))


    # 相关的数据结构，CPU吞吐
    epoch_train_time  = []      # 每个epoch的训练时间
    epoch_valid_time  = []      # 每个epoch的验证时间
    his_valid_m_loss  = []
    min_valid_m_loss  = 999999
    train_valid_result= []      # 训练-验证的数据存储
    min_test_m_loss   = 999999
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
                     args.learning_rate, args.weight_decay, set_data.scale)
    # GPU上的损失计算函数，用于循环里的模型直接测试
    evaluatel1 = nn.L1Loss (size_average=False).to(device)
    evaluatel2 = nn.MSELoss(size_average=False).to(device)



    '''
    ### (训练-验证-测试)大循环
    for i in range(1, args.epochs+1) :  # 10000 epochs


        # 清理GPU显存
        # report_gpu()


        ############ 1-epoch训练 ############
        train_total_loss = 0
        train_n_samples  = 0
        t1 = time.time()
        for iter, (x, y) in enumerate(set_data.get_batches(
            set_data.train_X, set_data.train_Y, batch_size=args.batch_size)) :
            # [B, W, N, 3] + [B, N]
            # 单步预测: [B, N]为第Horizons步的结果

            x = torch.Tensor(x).to(device)     # -- [B, W, N, 3]
            x = x.transpose(1, 3)              # -- [B, 3, N, W]
            y = torch.Tensor(y).to(device)     # -- [B, N]

            metrics = engine.train(x, y)       # -- loss

            train_total_loss += metrics                # loss acc
            train_n_samples  += (y.size(0)*y.size(1))  # B*N

            # if iter % args.cnt_log == 0 :
            #     log = 'Iter: {:04d}, Tr Loss: {:.4f}'
            #     print(log.format(iter, metrics / (y.size(0)*y.size(1))), flush=True)
        t2 = time.time()
        epoch_train_time.append(t2-t1)


        ############ 2-epoch验证 ############
        valid_total_loss_l1 = 0
        valid_total_loss_l2 = 0
        valid_n_samples     = 0
        valid_preds         = None
        valid_reals         = None
        v1 = time.time()
        for iter, (x, y) in enumerate(set_data.get_batches(
            set_data.valid_X, set_data.valid_Y, batch_size=args.batch_size)) :
            # [B, W, N, 3] + [B, N]
            # 单步预测: [B, N]为第Horizons步的结果

            x = torch.Tensor(x).to(device)     # -- [B, W, N, 3]
            x = x.transpose(1, 3)              # -- [B, 3, N, W]
            y = torch.Tensor(y).to(device)     # -- [B, N]

            metrics = engine.eval(x, y)

            valid_total_loss_l1 += metrics[0]  # R
            valid_total_loss_l2 += metrics[1]  # R
            valid_n_samples     += metrics[2]  # B*N
            if valid_preds is None :  # 首次进入
                valid_preds      = metrics[3]  # [B, N]
                valid_reals      = y           # [B, N]
            else :                             # -- [n*B, N]
                valid_preds = torch.cat((valid_preds, metrics[3]))
                valid_reals = torch.cat((valid_reals, y))

        # 验证集的指标
        valid_rae, valid_rse, valid_corr = \
            _1_util.metric(set_data, valid_total_loss_l1, valid_total_loss_l2, 
                           valid_n_samples, valid_preds, valid_reals)
        v2 = time.time()
        epoch_valid_time.append(v2-v1)


        ############ 3-epoch数据处理 ############
        train_m_loss = train_total_loss / train_n_samples  # 本轮epoch训练的平均loss
        his_valid_m_loss.append(valid_rse.item())          # 历史epoch验证的rse
        # 存储形式
        epoch_m_value = dict(
            epoch_train_m_loss = train_m_loss, 
            epoch_valid_m_rae  = valid_rae, 
            epoch_valid_m_rse  = valid_rse, 
            epoch_valid_m_corr = valid_corr
        )
        epoch_m_value = pd.Series(epoch_m_value)  # 转化为表的一行，类一维数组
        train_valid_result.append(epoch_m_value)  # epoch内的平均值大集合

        # 打印
        log = 'Epoch: {:04d}, Tr Time: {:.4f} secs, Vd Time: {:.4f} secs'
        print(log.format(i, (t2-t1), (v2-v1)))
        log = 'Tr Loss: {:.4f}, Vd RAE: {:.4f}, Vd RSE: {:.4f}, Vd CORR: {:.4f}'
        print(log.format(train_m_loss, valid_rae, valid_rse, valid_corr), flush=True)


        ############ 4-epoch测试 ############
        # 训练时model参数的优化方向，为使训练集loss下降的方向
        # 首先判断，优化后的model，能否使得验证集loss下降
        # 再次判断，优化后的model，能否使得测试集loss下降
        # 三集下降，则认为模型有效
        if valid_rse < min_valid_m_loss :  # 本轮验证集loss下降有效

            if i < 10 :  # 百轮之内，模型大概率未收敛，只要降低了验证集mae，就认为模型有效

                bestid           = i
                min_valid_m_loss = valid_rse

                log = 'Epoch: {:04d}, Valid Good!!! Valid_RSE: {:.4f}'
                print(log.format(i, min_valid_m_loss), flush=True)

                torch.save(engine.model.state_dict(), save_path + 'best_model.pth')
                cnt_updated = 0


            elif i >= 10 :  # 百轮之后，继续进行测试集有效判断

                test_total_loss_l1 = 0
                test_total_loss_l2 = 0
                test_n_samples     = 0
                test_preds         = None
                test_reals         = None

                for iter, (x, y) in enumerate(set_data.get_batches(
                    set_data.test_X, set_data.test_Y, batch_size=args.batch_size, shuffle=False)) :
                    # [B, W, N, 3] + [B, N]
                    # 单步预测: [B, N]为第Horizons步的结果

                    x = torch.Tensor(x).to(device)     # -- [B, W, N, 3]
                    x = x.transpose(1, 3)              # -- [B, 3, N, W]
                    y = torch.Tensor(y).to(device)     # -- [B, N]

                    engine.model.eval()
                    with torch.no_grad() :
                        preds, _ = engine.model(x)
                    preds = preds[:, :, :, -1].squeeze()

                    # device之上运算
                    scale = set_data.scale.expand(y.size(0), -1)
                    preds = preds * scale
                    y     = y     * scale

                    test_total_loss_l1 += evaluatel1(preds, y).item()
                    test_total_loss_l2 += evaluatel2(preds, y).item()
                    test_n_samples     += (y.size(0) * y.size(1))
                    if test_preds is None :  # 首次进入
                        test_preds = preds
                        test_reals = y
                    else :
                        test_preds = torch.cat((test_preds, preds))
                        test_reals = torch.cat((test_reals, y))

                # 测试集的指标
                test_rae, test_rse, test_corr = _1_util.metric(
                    set_data, test_total_loss_l1, test_total_loss_l2, test_n_samples, test_preds, test_reals)

                # 存储形式  # 代码可以下移，做成单调递减记录
                epoch_m_value = dict(
                    epoch_test_m_rae  = test_rae, 
                    epoch_test_m_rse  = test_rse, 
                    epoch_test_m_corr = test_corr
                )
                epoch_m_value = pd.Series(epoch_m_value)
                test_result.append(epoch_m_value)

                # 打印
                log = 'Epoch: {:04d}, Tt RAE: {:.4f}, Tt RSE: {:.4f}, Tt CORR: {:.4f}'
                print(log.format(i, test_rae, test_rse, test_corr), flush=True)


                # 判断本轮的测试指标，是否使得测试集loss下降
                if test_rse < min_test_m_loss :

                    bestid           = i
                    min_valid_m_loss = valid_rse
                    min_test_m_loss  = test_rse

                    log = '################ Epoch: {:04d}, Test Good!!!' + \
                          'Valid_Loss: {:.4f}, Test_Loss: {:.4f} ################'
                    print(log.format(i, min_valid_m_loss, min_test_m_loss), flush=True)

                    torch.save(engine.model.state_dict(), save_path + 'best_model.pth')
                    cnt_updated = 0

                else :  # 验证集loss下降，但是测试集loss没有下降
                    cnt_updated += 1
                    print('No Update: valid good + test bad')

        else :  # 训练出来的模型，没有使得验证集loss下降
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
    print('Best RSE: ', str(round(his_valid_m_loss[bestid-1], 4)))

    # print(asizeof.asizeof(train_valid_result))  # 
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


    # 最终测试
    last_total_loss_l1 = 0
    last_total_loss_l2 = 0
    last_n_samples     = 0
    last_preds         = None
    last_reals         = None
    # As                 = []  ##

    for iter, (x, y) in enumerate(set_data.get_batches(
        set_data.test_X, set_data.test_Y, batch_size=args.batch_size, shuffle=False)) :
        # [B, W, N, 3] + [B, N]
        # 单步预测: [B, N]为第Horizons步的结果

        x = torch.Tensor(x).to(device)     # -- [B, W, N, 3]
        x = x.transpose(1, 3)              # -- [B, 3, N, W]
        y = torch.Tensor(y).to(device)     # -- [B, N]

        engine.model.eval()
        with torch.no_grad() :
            preds, A = engine.model(x)
        preds = preds[:, :, :, -1].squeeze()

        # device之上运算
        scale = set_data.scale.expand(y.size(0), -1)
        preds = preds * scale
        y     = y     * scale

        last_total_loss_l1 += evaluatel1(preds, y).item()
        last_total_loss_l2 += evaluatel2(preds, y).item()
        last_n_samples     += (y.size(0) * y.size(1))
        if last_preds is None :  # 首次进入
            last_preds = preds
            last_reals = y
        else :
            last_preds = torch.cat((last_preds, preds))
            last_reals = torch.cat((last_reals, y))
        # As.append(A)
    # A_ = torch.cat(As, dim=0)


    # 指标计算
    last_rae, last_rse, last_corr = _1_util.metric(
        set_data, last_total_loss_l1, last_total_loss_l2, last_n_samples, last_preds, last_reals)


    # 打印
    log = 'Tt RAE: {:.4f}, Tt RSE: {:.4f}, Tt CORR: {:.4f}'
    print(log.format(last_rae, last_rse, last_corr), flush=True)



    ############ 模型测试的记录 ############


    # 参数回顾
    print('seed: {:04d}'.format(seed))
    print(args)


    # 测试文件存储
    test_m_value = []
    test_m_value = dict(
        test_m_rae  = last_rae, 
        test_m_rse  = last_rse, 
        test_m_corr = last_corr
    )
    test_m_value = pd.Series(test_m_value)
    test_result.append(test_m_value)
    # print(asizeof.asizeof(test_result))  # 

    test_2_csv = pd.DataFrame(test_result)
    test_2_csv.round(8).to_csv(f'{save_path}/test.csv')


    # 保存相关画图数据为.npy
    np.save(save_path + 'y_real.npy', last_reals.detach().cpu().numpy())
    np.save(save_path + 'y_hat.npy',  last_preds.detach().cpu().numpy())
    # np.save(save_path + 'As.npy',     A_        .detach().cpu().numpy())

