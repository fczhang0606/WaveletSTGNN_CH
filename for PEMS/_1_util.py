########################
####### 20250402 #######
########################



# 
import torch

import numpy as np
import os
from pympler import asizeof



###### 分集，制成批函数 ######
class DataLoader(object) :


    def __init__(self, x, y, batch_size, pad_with_last_sample=True) :

        self.batch_size  = batch_size
        self.batch_ind   = 0

        if pad_with_last_sample :

            num_padding = (batch_size - (len(x) % batch_size)) % batch_size  # 53 + 10699 = 64*168

            # np.()
            x_padding   = np.repeat(x[-1:], num_padding, axis=0)  # Repeat the last sample
            # print(x_padding)  # (53, 12, 170, 3)
            y_padding   = np.repeat(y[-1:], num_padding, axis=0)  # 
            # print(y_padding)  # (53, 12, 170, 1)

            x = np.concatenate([x, x_padding], axis=0)
            y = np.concatenate([y, y_padding], axis=0)

        ### 长数据的规整
        self.x         = x
        self.y         = y
        self.size      = len(x)  # 10752
        self.num_batch = int(self.size // self.batch_size)  # 168


    def shuffle(self) :

        permutation = np.random.permutation(self.size)  # 
        x, y        = self.x[permutation], self.y[permutation]
        self.x      = x
        self.y      = y


    # for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()) :
    def get_iterator(self) :  # 调用一次，即可返回168批

        self.batch_ind = 0  # 批号

        def _wrapper() :  # 内嵌的装饰器函数，拓展功能

            while self.batch_ind < self.num_batch :

                start_ind = self.batch_size*self.batch_ind
                end_ind   = min(self.size, self.batch_size*(self.batch_ind+1))
                x_i = self.x[start_ind:end_ind, ...]
                y_i = self.y[start_ind:end_ind, ...]
                yield (x_i, y_i)  # 类似return，节约内存

                self.batch_ind += 1

        return _wrapper()  # 返回函数



###### 缩放函数 ######
class StandardScaler :


    def __init__(self, mean, std) :
        self.mean = mean  # 均值
        self.std  = std   # 标准差


    def transform(self, data) :
        return (data - self.mean) / self.std


    def inverse_transform(self, data) :
        return (data * self.std) + self.mean



###### 主调用函数，./dataset/PEMS08/(train/valid/test).npz ######
def load_dataset(dataset_dir, train_batch_size, valid_batch_size=None, test_batch_size=None) :


    data = {}  # 字典，CPU存储，函数返回
    # print(asizeof.asizeof(data))  # 64


    # 填充字典的键值对
    for category in ['train', 'valid', 'test'] :

        raw_data = np.load(os.path.join(dataset_dir, category+'.npz'))  # .npz是NumPy库用来存储多个NumPy数组

        # [L, W, N, C]
        data['x_' + category] = raw_data['x']
        # print(data['x_' + category].shape)  # (10699/3567/3567, 12, 170, 3)
        data['y_' + category] = raw_data['y']
        # print(data['y_' + category].shape)  # (10699/3567/3567, 12, 170, 1)
    # print(asizeof.asizeof(data))  # 1164139912/(17833*12*170*4) = 1164139912/145517280 = 8 bytes (float)


    # 字典数据处理：三部x的归一化，后续对预测的prediction进行了反归一化，y_real没有归一化
    scaler = StandardScaler(  # https://blog.csdn.net/g944468183/article/details/124473886
        mean = data['x_train'][..., 0].mean(),  # 最后一维是features的值，均以训练集的值为基准
        std  = data['x_train'][..., 0].std()
    )

    for category in ['train', 'valid', 'test'] :
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])


    # 字典随机重排    # torch.()是否会自动将np转为tensor？应该不会
    random_train    = torch.arange(int(data['x_train'].shape[0]))
    random_train    = torch.randperm(random_train.size(0))
    data['x_train'] = data['x_train'][random_train, ...]
    data['y_train'] = data['y_train'][random_train, ...]

    random_valid    = torch.arange(int(data['x_valid'].shape[0]))
    random_valid    = torch.randperm(random_valid.size(0))
    data['x_valid'] = data['x_valid'][random_valid, ...]
    data['y_valid'] = data['y_valid'][random_valid, ...]

    # random_test     = torch.arange(int(data['x_test'].shape[0]))
    # random_test     = torch.randperm(random_test.size(0))
    # data['x_test']  = data['x_test'][random_test, ...]
    # data['y_test']  = data['y_test'][random_test, ...]


    # data组织四个键值对，后续取出批次batch
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], train_batch_size)
    data['valid_loader'] = DataLoader(data['x_valid'], data['y_valid'], valid_batch_size)
    data['test_loader']  = DataLoader(data['x_test'] , data['y_test'] , test_batch_size)
    data['scaler']       = scaler  # 两个值 + 两个函数
    # print(asizeof.asizeof(data))  # 2331871240


    # data的冗余数据，删除
    del data['x_train']
    del data['y_train']
    del data['x_valid']
    del data['y_valid']
    del data['x_test']
    # del data['y_test']
    # print(asizeof.asizeof(data))  # 1167731704


    return data



def MAE_torch  (pred, real, mask_value=None) :

    if mask_value != None :
        mask = torch.gt(real, mask_value)  # ge/gt/le/lt/ne/eq分别是>=/>/<=/</==/!=
        pred = torch.masked_select(pred, mask)
        real = torch.masked_select(real, mask)

    return torch.mean(torch.abs(pred - real))


def MAPE_torch (pred, real, mask_value=None) :

    if mask_value != None :
        mask = torch.gt(real, mask_value)  # ge/gt/le/lt/ne/eq分别是>=/>/<=/</==/!=
        pred = torch.masked_select(pred, mask)
        real = torch.masked_select(real, mask)

    return torch.mean(torch.abs(torch.div((pred - real), real)))


def RMSE_torch (pred, real, mask_value=None) :

    if mask_value != None :
        mask = torch.gt(real, mask_value)  # ge/gt/le/lt/ne/eq分别是>=/>/<=/</==/!=
        pred = torch.masked_select(pred, mask)
        real = torch.masked_select(real, mask)

    return torch.sqrt(torch.mean((pred - real) ** 2))


def WMAPE_torch(pred, real, mask_value=None) :

    if mask_value != None :
        mask = torch.gt(real, mask_value)  # ge/gt/le/lt/ne/eq分别是>=/>/<=/</==/!=
        pred = torch.masked_select(pred, mask)
        real = torch.masked_select(real, mask)

    return torch.sum(torch.abs(pred - real)) / torch.sum(torch.abs(real))


def metric(pred, real) :

    mae   = MAE_torch  (pred, real, 0.0).item()
    mape  = MAPE_torch (pred, real, 0.0).item()
    rmse  = RMSE_torch (pred, real, 0.0).item()
    wmape = WMAPE_torch(pred, real, 0.0).item()

    return mae, mape, rmse, wmape

