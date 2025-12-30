########################
####### 20250402 #######
########################



# 
import torch
from torch.autograd import Variable

import math
import numpy as np
from pympler import asizeof



class load_dataset(object) :


    # 变量管理
    # self.scale                    # torch.(cuda:0)
    # self.train_X + self.train_Y   # torch.CPU
    # self.valid_X + self.valid_Y   # torch.CPU
    # self.test_X  + self.test_Y    # torch.CPU
    # self.rae     + self.rse       # float
    def __init__(self, device, dataset_dir, train_ratio, valid_ratio, windows, horizons, granularity) :


        fin     = open(dataset_dir + '.txt')        # 文件句柄
        rawdata = np.loadtxt(fin, delimiter=',')    # 原始数据
        # if rawdata.shape[1] == 862 :  # traffic
        #     index   = np.random.choice(862, 462, replace=False)  # 选择462列删除
        #     rawdata = np.delete(rawdata, index, axis=1)
        # print(asizeof.asizeof(rawdata))  # 

        newdata    = np.zeros(rawdata.shape)        # 待刷矩阵
        L, N       = newdata.shape                  # [L, N]
        self.scale = np.ones(N)                     # 列(每个节点)缩放向量，[N]
        for i in range(N) :
            self.scale[i] = np.max(np.abs(rawdata[:, i]))  # 列(每个节点)缩放因子
            newdata[:, i] = rawdata[:, i] / self.scale[i]  # 列(每个节点)值被刷新
        # print(asizeof.asizeof(newdata))  # 


        # 位置嵌入向量
        ftr = newdata                   # [L, N]
        ftr = ftr[:, :, np.newaxis]     # [L, N, 1]

        # stamp_list[0]
        stamp_list  = [ftr]             # [[L, N, 1]]

        # stamp_list[1]
        time_in_day = [i%granularity/granularity for i in range(L)]         # 取余取除，浮点小数的list
        time_in_day = np.array(time_in_day)                                 # [L, ]
        time_in_day = np.tile(time_in_day, [1, N, 1]).transpose((2, 1, 0))  # (1, N, L) -- (L, N, 1)
        # https://blog.csdn.net/PSpiritV/article/details/123266458
        stamp_list.append(time_in_day)

        # stamp_list[2]
        day_in_week = [(i//granularity) % 7      for i in range(L)]         # 向下取整 + 取余
        day_in_week = np.array(day_in_week)
        day_in_week = np.tile(day_in_week, [1, N, 1]).transpose((2, 1, 0))
        stamp_list.append(day_in_week)

        # stamp_list(L, N, 3) ---- newdata
        newdata = np.concatenate(stamp_list, axis=-1)                       # -- (L, N, 3)
        # print(asizeof.asizeof(newdata))  #


        # 从newdata划分(训练、验证、测试)三集，得到六个部分
        self._split(newdata, int(train_ratio*L), int((train_ratio+valid_ratio)*L), L, 
                    windows, horizons)
        # set_: [M, W, N, 3] + [M, N]


        # scale + rae + rse
        self.scale = torch.from_numpy(self.scale).float()           # [N]

        # Hadamard: [M, N] * [M, N] = [M, N]  # 缩放数据还原，反归一化
        # https://blog.csdn.net/weixin_38346042/article/details/138586605
        _test_y    = self.test_Y * self.scale.expand(self.test_Y.size(0), N)

        # https://blog.csdn.net/allein_STR/article/details/128678952
        self.rae   = torch.mean(torch.abs(_test_y - torch.mean(_test_y)))
        self.rse   = _test_y.std() * np.sqrt( (len(_test_y) - 1.)/(len(_test_y)) )
        print('rae:{}'.format(self.rae))
        print('rse:{}'.format(self.rse))

        self.scale = self.scale.to(device)


    def _split(self, newdata, train_size, tra_val_size, L, windows, horizons) :

        # 分集顺序序号
        train_idx = range(windows+horizons-1, train_size)   # windows+horizons为一条数据的长度
        valid_idx = range(train_size,       tra_val_size)
        test_idx  = range(tra_val_size,                L)

        # 返回self.全局的torch数据[M, W, N, 3] + [M, N]
        self.train_X, self.train_Y = self._batchify(newdata, train_idx, windows, horizons)
        self.valid_X, self.valid_Y = self._batchify(newdata, valid_idx, windows, horizons)
        self.test_X,  self.test_Y  = self._batchify(newdata, test_idx,  windows, horizons)

        # 形状
        # print('train: {} {}'.format(self.train_X.shape, self.train_Y.shape))
        # print('valid: {} {}'.format(self.valid_X.shape, self.valid_Y.shape))
        # print('test:  {} {}'.format(self.test_X.shape,  self.test_Y.shape ))

        # 大小
        # print(self.train_X.element_size() * self.train_X.nelement())
        # print(self.train_Y.element_size() * self.train_Y.nelement())
        # print(self.valid_X.element_size() * self.valid_X.nelement())
        # print(self.valid_Y.element_size() * self.valid_Y.nelement())
        # print(self.test_X.element_size()  * self.test_X.nelement())
        # print(self.test_Y.element_size()  * self.test_Y.nelement())


    def _batchify(self, newdata, set_idx, windows, horizons) :  # idx: 数据条的终点的位置序列

        N    = newdata.shape[1]  # 节点数
        nums = len(set_idx)      # 分集的数据条的数量

        # 存储区域：是否有更经济的存取结构？
        X = torch.zeros((nums, windows, N, 3))      # [M, W, N, 3] -- [M, C, N, W]
        Y = torch.zeros((nums, N))                  # [M, N]       -- [M, C, N, H]
        # Y = torch.zeros((nums, horizons, N, 1))     # [M, H, N, 1] -- [M, C, N, H]

        for i in range(nums) :  # 每一条，填充[W, N, 3] + [N]

            # X[i]的起点、终点位置
            end   = set_idx[i] + 1 - horizons   # windows的下一点，本地点的位置，不在X[i]内
            start = end - windows               # windows的起点

            # 读取newdata(L, N, 3)数据: numpy ---- torch
            X[i, :, :, :] = torch.from_numpy(newdata[start:end , :, :])  # [W, N, 3] -- 左闭右开
            Y[i, :]       = torch.from_numpy(newdata[set_idx[i], :, 0])  # [N]       -- horizons包含本地点和预测点

        return X, Y  # [M, W, N, 3] + [M, N]



    # [M, W, N, 3] + [M, N]
    def get_batches(self, set_X, set_Y, batch_size, shuffle=True) :

        nums = len(set_X)  # 分集的数据条的数量，没有paddings补齐

        if shuffle :
            index = torch.randperm(nums)           # 定长乱序的编号序列
        else :
            index = torch.LongTensor(range(nums))  # 定长顺序

        batch_start = 0  # 变量：新的batch的起始序号
        while (batch_start < nums) :
            batch_end = min(batch_start + batch_size, nums)
            excerpt   = index[batch_start:batch_end]    # 乱序编号序列的部分节选，仍是乱序编号序列
            X         = set_X[excerpt]
            Y         = set_Y[excerpt]
            yield X, Y                                  # [B, W, N] + [B, N]
            batch_start += batch_size



def metric(data, loss_l1, loss_l2, samples, preds, reals) :

    rae = (loss_l1/samples) / data.rae
    rse = math.sqrt(loss_l2 / samples) / data.rse

    p   = preds.data.cpu().numpy()  # [L, N]
    y   = reals.data.cpu().numpy()  # [L, N]

    sigma_p = p.std (axis=0)
    sigma_y = y.std (axis=0)

    mean_p  = p.mean(axis=0)
    mean_y  = y.mean(axis=0)

    index   = (sigma_y != 0)
    corr    = ((p-mean_p) * (y-mean_y)).mean(axis=0) / (sigma_p * sigma_y)
    corr    = corr[index].mean()

    return rae, rse, corr


# electricity       rae:3429.722900390625       rse:16619.361328125
# exchange_rate     rae:0.3432909846305847      rse:0.4557727873325348
# solar_AL          rae:6.59852409362793        rse:8.921234130859375
# traffic           rae:0.03793337568640709     rse:0.057082220911979675

