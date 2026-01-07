####
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


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
    node = 97                   ################ 选择节点 ################
    data_1 = data_1[:, node, 0] # (16992,)
    data_2 = data_2[:, node, 0] # (3394,)
    data_3 = data_3[:, node, 0] # (3375,)
    data_4 = data_4[:, node, 0] # (3375,)
    data_5 = data_5[:, node, 0] # (3394,)
    data_6 = data_6[:, node, 0] # (3394,)
    data_7 = data_7[:, node, 0] # (3394,)
    data_8 = data_8[:, node, 0] # (3394,)
################################################################
    # PEMS04
    # January 1, 2018 to February 28, 2018 = 59 days = 59*288 = 16992

    # 2018.02.18
    # 0101～0217 = 48 days = 48*288 = 13824
    # 13824-13606 = 218
    p = 218         # 02.18     ################ 选择时间 ################
    d = 6           # (16992-13824)/288=11, [0, 10)
    l = 288         # 一日长度

    p = 0
    d = 0
    l = 3375

    data_1_ = data_1[13606+p+d*288:13606+p+d*288+l]     # 全局real的天内片段
    data_2_ = data_2[19+p+d*288   :19+p+d*288+l]        # 测试集real的天内片段

    data_3_ = data_3[p+d*288      :p+d*288+l]           # STWave-real的天内片段
    data_5_ = data_5[19+p+d*288   :19+p+d*288+l]        # STIDGCN-real的天内片段
    data_7_ = data_7[19+p+d*288   :19+p+d*288+l]        # DWISTGNN-real的天内片段

    data_4_ = data_4[p+d*288      :p+d*288+l]           # STWave-pred的天内片段
    data_6_ = data_6[19+p+d*288   :19+p+d*288+l]        # STIDGCN-pred的天内片段
    data_8_ = data_8[19+p+d*288   :19+p+d*288+l]        # DWISTGNN-pred的天内片段

    # 整合绘图数据
    data_sets = [data_2_,   # 测试集real的天内片段
                 data_4_,   # STWave-pred的天内片段
                 data_6_,   # STIDGCN-pred的天内片段
                 data_8_]   # DWISTGNN-pred的天内片段
################################################################
    plt.figure (figsize=(20, 12))

    # 散点图
    plt.scatter(data_2_, data_2_, alpha=0.3, label='Ground Truth')
    plt.scatter(data_3_, data_4_, alpha=0.3, label='STWave')
    plt.scatter(data_5_, data_6_, alpha=0.3, label='STIDGCN')
    plt.scatter(data_7_, data_8_, alpha=0.3, label='DWISTGNN')

    # 线性拟合
    slope_0, intercept_0, r_value_0, _, _ = stats.linregress(data_2_, data_2_)
    slope_1, intercept_1, r_value_1, _, _ = stats.linregress(data_3_, data_4_)
    slope_2, intercept_2, r_value_2, _, _ = stats.linregress(data_5_, data_6_)
    slope_3, intercept_3, r_value_3, _, _ = stats.linregress(data_7_, data_8_)

    fit_line_0 = slope_0 * data_2_ + intercept_0
    fit_line_1 = slope_1 * data_3_ + intercept_1
    fit_line_2 = slope_2 * data_5_ + intercept_2
    fit_line_3 = slope_3 * data_7_ + intercept_3


    # 绘制曲线
    plt.plot(data_2_, fit_line_0, lw=1, label=f"Fit: y={slope_0:.4f}x+{intercept_0:.4f}")
    print(f"{'Ground Truth'}:   y = {slope_0:.4f}x + {intercept_0:.4f} (R²={r_value_0**2:.4f})")

    plt.plot(data_3_, fit_line_1, lw=1, label=f"Fit: y={slope_1:.4f}x+{intercept_1:.4f}")
    print(f"{'STWave'}:         y = {slope_1:.4f}x + {intercept_1:.4f} (R²={r_value_1**2:.4f})")

    plt.plot(data_5_, fit_line_2, lw=1, label=f"Fit: y={slope_2:.4f}x+{intercept_2:.4f}")
    print(f"{'STIDGCN'}:        y = {slope_2:.4f}x + {intercept_2:.4f} (R²={r_value_2**2:.4f})")

    plt.plot(data_7_, fit_line_3, lw=1, label=f"Fit: y={slope_3:.4f}x+{intercept_3:.4f}")
    print(f"{'DWISTGNN'}:       y = {slope_3:.4f}x + {intercept_3:.4f} (R²={r_value_3**2:.4f})")


    # 绘制全图
    plt.title("PEMS04-Scatter Plots with Linear Fits", fontsize=36)
    plt.xlabel("Real Value - " + f"Node {node+1} in 2018.02.{18+d}", fontsize=36)
    plt.xlabel("Real Value - " + f"Node {node+1} from 2018.01.01 to 2018.02.28", fontsize=36)
    plt.ylabel("Predicted Value", fontsize=36)
    plt.legend(fontsize=20)
    plt.grid(alpha=0.3)


    # 保存图像
    save_path = "fig_3_regression_04.png"
    save_path = "fig_3_regression_04 (3375).png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"✅ 图表已保存至: {save_path}")
################################################################

