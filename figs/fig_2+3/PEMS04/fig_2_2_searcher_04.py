####
import numpy as np


if __name__ == "__main__" :


################################################################
    # 数据文件
    file_1 = './data/PEMS04.npz'            # 总数据集
    file_2 = './data/PEMS04_test.npz'       # 测试集
    file_3 = './data/STWave_y_real.npy'     # STWave-real
    file_4 = './data/STWave_y_hat.npy'      # STWave-pred       # 换参数测试
    file_5 = './data/STIDGCN_y_real.npy'    # STIDGCN-real
    file_6 = './data/STIDGCN_y_hat.npy'     # STIDGCN-pred      # 换参数测试
    file_7 = './data/DWISTGNN_y_real.npy'   # DWISTGNN-real
    file_8 = './data/DWISTGNN_y_hat.npy'    # DWISTGNN-pred     # 最优参数测试

    # 数据读取
    data_1 = np.load(file_1)    # ['data']
    data_1 = data_1['data']     # (16992, 307, 3)
    data_2 = np.load(file_2)    # ['x', 'y', 'x_offsets', 'y_offsets']
    data_2 = data_2['y']        # (3394, 12, 307, 1)
    data_2 = np.swapaxes(data_2, 1, 3)[:, 0, :, :]  # (3394, 307, 12)
    data_3 = np.load(file_3)    # (3375, 307, 12)
    data_4 = np.load(file_4)    # (3375, 307, 12)
    data_5 = np.load(file_5)    # (3394, 307, 12)
    data_6 = np.load(file_6)    # (3394, 307, 12)
    data_7 = np.load(file_7)    # (3394, 307, 12)
    data_8 = np.load(file_8)    # (3394, 307, 12)


    # 数据划分
    node = 0                    ################ 选择节点 ################
    # PEMS04
    # January 1, 2018 to February 28, 2018 = 59 days = 59*288 = 16992
    # 2018.02.18
    # 0101～0217 = 48 days = 48*288 = 13824
    # 13824-13606 = 218
    p = 218         # 02.18     ################ 选择时间 ################
    d = 0           # (16992-13824)/288=11, [0, 10], 16992取不到所以实际上取不到10
    h = 0           # [0, 23]
    ld = 288        # 一日长度
    lh = 12         # 一时长度


    errors = []
    for i in range(307) :                       # 节点遍历
        data_2_ = data_2[:, i, 0] # (3394,)
        data_4_ = data_4[:, i, 0] # (3375,)
        data_6_ = data_6[:, i, 0] # (3394,)
        data_8_ = data_8[:, i, 0] # (3394,)

        for j in range(10) :                    # 天数遍历

            for k in range(24) :                # 小时遍历

                data_2__ = data_2_[19+p+j*ld+k*lh:19+p+j*ld+k*lh+3*lh]  # 3个小时的窗口
                data_4__ = data_4_[   p+j*ld+k*lh:   p+j*ld+k*lh+3*lh]
                data_6__ = data_6_[19+p+j*ld+k*lh:19+p+j*ld+k*lh+3*lh]
                data_8__ = data_8_[19+p+j*ld+k*lh:19+p+j*ld+k*lh+3*lh]

                error_1 = np.mean(np.abs(data_2__ - data_4__))          # G - STWave
                error_2 = np.mean(np.abs(data_2__ - data_6__))          # G - STIDGCN
                error_3 = np.mean(np.abs(data_2__ - data_8__))          # G - DWISTGNN

                if error_1 > error_3 and error_2 > error_3 :
                    if error_1 + error_2 - 2*error_3 > 10 :
                        errors.append([i, j, k, error_1 + error_2 - 2*error_3])
    print(f"总共有 {len(errors)} 个误差向量。")


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

