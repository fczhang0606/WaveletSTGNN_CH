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
    data_1 = data_1[:, node, 0] # (17856,)
    data_2 = data_2[:, node, 0] # (3567,)
    data_3 = data_3[:, node, 0] # (3548,)
    data_4 = data_4[:, node, 0] # (3548,)
    data_5 = data_5[:, node, 0] # (3567,)
    data_6 = data_6[:, node, 0] # (3567,)
    data_7 = data_7[:, node, 0] # (3567,)
    data_8 = data_8[:, node, 0] # (3567,)
################################################################
    # 查看3567在17856中的匹配位置，有17856-3567+1=14290个段
    for i in range(17856-3567+1) :          # [0, 14289]
        data_window = data_1[i:i+3567-1+1]  # [0, 3566]  # 数据截取，左开右闭

        # 段内3567个元素匹配
        cnt = 0
        for j in range(3567) :
            if np.abs(data_window[j] - data_2[j]) <= 1 :
                cnt += 1
        # 匹配度
        if cnt == 3567 :
            print(f"测试集匹配索引: {i}")       # 14278

    # 查看3548在17856中的匹配位置，有17856-3548+1=14309个段
    for i in range(17856-3548+1) :          # [0, 14308]
        data_window = data_1[i:i+3548-1+1]  # [0, 3547]  # 数据截取，左开右闭

        # 段内3548个元素匹配
        cnt = 0
        for j in range(3548) :
            if np.abs(data_window[j] - data_3[j]) <= 1 :
                cnt += 1
        # 匹配度
        if cnt == 3548 :
            print(f"STWave匹配索引: {i}")       # 14297

    # 查看3567在17856中的匹配位置，有17856-3567+1=14290个段
    for i in range(17856-3567+1) :          # [0, 14289]
        data_window = data_1[i:i+3567-1+1]  # [0, 3566]  # 数据截取，左开右闭

        # 段内3567个元素匹配
        cnt = 0
        for j in range(3567) :
            if np.abs(data_window[j] - data_5[j]) <= 1 :
                cnt += 1
        # 匹配度
        if cnt == 3567 :
            print(f"STIDGCN匹配索引: {i}")      # 14278

    # 查看3567在17856中的匹配位置，有17856-3567+1=14290个段
    for i in range(17856-3567+1) :          # [0, 14289]
        data_window = data_1[i:i+3567-1+1]  # [0, 3566]  # 数据截取，左开右闭

        # 段内3567个元素匹配
        cnt = 0
        for j in range(3567) :
            if np.abs(data_window[j] - data_7[j]) <= 1 :
                cnt += 1
        # 匹配度
        if cnt == 3567 :
            print(f"DWISTGNN匹配索引: {i}")      # 14278
################################################################
    # 查看3548在3567中的匹配位置，有3567-3548+1=20个段
    for i in range(3567-3548+1) :           # [0, 19]
        data_window = data_2[i:i+3548-1+1]  # [0, 3547]  # 数据截取，左开右闭

        # 段内3548个元素匹配
        cnt = 0
        for j in range(3548) :
            if np.abs(data_window[j] - data_3[j]) <= 1 :
                cnt += 1
        # 匹配度
        if cnt == 3548 :
            print(f"(3548/3567)匹配索引: {i}")  # 19
################################################################
    # 3548段内，数据截取与比对
    p = 66      # 起始位置
    l = 200     # 长度
    data_2_ = data_2[19+p:19+p+l]   # 测试集-real
    data_3_ = data_3[p:p+l]         # STWave-real
    data_5_ = data_5[19+p:19+p+l]   # STIDGCN-real
    data_7_ = data_7[19+p:19+p+l]   # DWISTGNN-real

    # real的匹配
    cnt = 0
    for k in range(l) :
        if np.abs(data_2_[k] - data_3_[k]) <= 1 :
            cnt += 1
    # 匹配度
    if cnt == l :
        print(f"测试集-real 和 STWave-real 匹配")
################################################################

