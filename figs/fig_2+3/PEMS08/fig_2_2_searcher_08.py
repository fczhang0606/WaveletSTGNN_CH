####
import numpy as np


if __name__ == "__main__" :


################################################################
    # 数据文件
    file_1 = './data/PEMS08.npz'            # 总数据集
    file_2 = './data/PEMS08_test.npz'       # 测试集
    file_3 = './data/STWave_y_real.npy'     # STWave-real
    file_4 = './data/STWave_y_hat.npy'      # STWave-pred       # 换参数测试
    file_5 = './data/STIDGCN_y_real.npy'    # STIDGCN-real
    file_6 = './data/STIDGCN_y_hat.npy'     # STIDGCN-pred      # 换参数测试
    file_7 = './data/DWISTGNN_y_real.npy'   # DWISTGNN-real
    file_8 = './data/DWISTGNN_y_hat.npy'    # DWISTGNN-pred     # 最优参数测试

    # 数据读取
    data_1 = np.load(file_1)    # ['data']
    data_1 = data_1['data']     # (17856, 170, 3)
    data_2 = np.load(file_2)    # ['x', 'y', 'x_offsets', 'y_offsets']
    data_2 = data_2['y']        # (3567, 12, 170, 1)
    data_2 = np.swapaxes(data_2, 1, 3)[:, 0, :, :]  # (3567, 170, 12)
    data_3 = np.load(file_3)    # (3548, 170, 12)
    data_4 = np.load(file_4)    # (3548, 170, 12)
    data_5 = np.load(file_5)    # (3567, 170, 12)
    data_6 = np.load(file_6)    # (3567, 170, 12)
    data_7 = np.load(file_7)    # (3567, 170, 12)
    data_8 = np.load(file_8)    # (3567, 170, 12)


    # 数据划分
    node = 0                    ################ 选择节点 ################
    # PEMS08
    # July 1, 2016 to August 31, 2016 = 62 days = 62*288 = 17856
    # 2016.08.20
    # 0701～0819 = 50 days = 50*288 = 14400
    # 14400-14297 = 103
    p = 103         # 08.20     ################ 选择时间 ################
    d = 0           # (17855-14400+1)/288=12, [0, 11]
    h = 0           # [0, 23]
    ld = 288        # 一日长度
    lh = 12         # 一时长度


    errors = []
    for i in range(170) :
        data_2_ = data_2[:, i, 0] # (3567,)
        data_4_ = data_4[:, i, 0] # (3548,)
        data_6_ = data_6[:, i, 0] # (3567,)
        data_8_ = data_8[:, i, 0] # (3567,)

        for j in range(12) :

            for k in range(24) :

                data_2__ = data_2_[19+p+j*ld+k*lh:19+p+j*ld+k*lh+3*lh]
                data_4__ = data_4_[   p+j*ld+k*lh:   p+j*ld+k*lh+3*lh]
                data_6__ = data_6_[19+p+j*ld+k*lh:19+p+j*ld+k*lh+3*lh]
                data_8__ = data_8_[19+p+j*ld+k*lh:19+p+j*ld+k*lh+3*lh]

                error_1 = np.mean(np.abs(data_2__ - data_4__))
                error_2 = np.mean(np.abs(data_2__ - data_6__))
                error_3 = np.mean(np.abs(data_2__ - data_8__))

                if error_1 > error_3 and error_2 > error_3 :
                    if error_1 + error_2 - 2*error_3 > 20 :
                        errors.append([i, j, k, error_1 + error_2 - 2*error_3])
    print(f"总共有 {len(errors)} 个错误向量。")


    indexs = []
    sorted_with_index = sorted(
        enumerate(errors), 
        key=lambda x: x[1][-1], 
        reverse=True)
    for orig_idx, vec in sorted_with_index :
    	if (vec[0], vec[1]) not in indexs :
            indexs.append((vec[0], vec[1]))
    print(indexs)
# ################################################################

