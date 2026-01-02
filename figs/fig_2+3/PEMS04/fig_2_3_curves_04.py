####
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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


    # PEMS04
    # January 1, 2018 to February 28, 2018 = 59 days = 59*288 = 16992
    # 2018.02.18
    # 0101～0217 = 48 days = 48*288 = 13824
    # 13824-13606 = 218
    p = 218         # 02.18     ################ 选择时间 ################
    l = 288         # 一日长度


    # searcher的结果
    rec = [(3, 4)]
    for node, d in rec :

        data_1_ = data_1[:, node, 0] # (16992,)
        data_2_ = data_2[:, node, 0] # (3394,)
        data_3_ = data_3[:, node, 0] # (3375,)
        data_4_ = data_4[:, node, 0] # (3375,)
        data_5_ = data_5[:, node, 0] # (3394,)
        data_6_ = data_6[:, node, 0] # (3394,)
        data_7_ = data_7[:, node, 0] # (3394,)
        data_8_ = data_8[:, node, 0] # (3394,)


        data_1__ = data_1_[13606+p+d*288:13606+p+d*288+l]       # 全局real的天内片段
        data_2__ = data_2_[19+p+d*288   :19+p+d*288+l]          # 测试集real的天内片段

        data_3__ = data_3_[p+d*288      :p+d*288+l]             # STWave-real的天内片段
        data_5__ = data_5_[19+p+d*288   :19+p+d*288+l]          # STIDGCN-real的天内片段
        data_7__ = data_7_[19+p+d*288   :19+p+d*288+l]          # DWISTGNN-real的天内片段

        data_4__ = data_4_[p+d*288      :p+d*288+l]             # STWave-pred的天内片段
        data_6__ = data_6_[19+p+d*288   :19+p+d*288+l]          # STIDGCN-pred的天内片段
        data_8__ = data_8_[19+p+d*288   :19+p+d*288+l]          # DWISTGNN-pred的天内片段


        # 整合绘图数据
        data_sets = [data_2__,  # 测试集real的天内片段
                     data_4__,  # STWave-pred的天内片段
                     data_6__,  # STIDGCN-pred的天内片段
                     data_8__]  # DWISTGNN-pred的天内片段


    ################################################################
        # 多子图
        fig, ax = plt.subplots(figsize=(20, 12))
        fig.patch.set_facecolor('white')  # 白色背景

        # 样式：标签、颜色、线型
        labels  = ["Ground Truth", "STWave", "STIDGCN", "DWISTGNN"]
        colors  = ["#3205e8", "#0D0E01", "#047D18", "#d80909"]
        styles  = ['-', '--', '-.', ':']


        ax.set_title('Model Comparison on PEMS04', fontsize=36)

        ax.set_xlabel(f'Node {node+1} in February {18+d}, 2018', 
                      fontsize=36)              #### 选择 ####
        #### x轴刻度默认为一天 ####
        ax.set_xlim(0, 288)                     # [0, 287]              = 288个点
        ax.set_xticks(np.arange(0, 300, 12))    # [0, 12, 24, ..., 288] = 24段（25个刻度）
        ax.set_xticklabels(['00:00', '01:00', '02:00', '03:00', '04:00', 
                            '05:00', '06:00', '07:00', '08:00', '09:00', 
                            '10:00', '11:00', '12:00', '13:00', '14:00', 
                            '15:00', '16:00', '17:00', '18:00', '19:00', 
                            '20:00', '21:00', '22:00', '23:00', '24:00'])

        ax.set_ylabel('Traffic Flow', fontsize=36)
        ax.set_ylim(np.min(data_sets) - (np.max(data_sets)-np.min(data_sets))/5, 
                    np.max(data_sets) + (np.max(data_sets)-np.min(data_sets))/5)

        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='gray')


        # 绘制数据曲线
        x = np.arange(0, 288, 1)        # (288,)
        for i in range(len(labels)) :   # 曲线数量
            ax.plot(x, data_sets[i], 
                    label       = labels[i], 
                    color       = colors[i], 
                    linestyle   = styles[i], 
                    linewidth   = 2)
        plt.legend(fontsize=30, framealpha=0.9, loc='best')
    ################################################################
        # 子窗口1
        plot1 = inset_axes(ax, 
                        width="80%", height="60%",               # 比例
                        loc='upper left', 
                        bbox_to_anchor=(0.38, -0.12, 0.5, 0.5),  # 位置+形变
                        bbox_transform=ax.transAxes)             # 从父坐标系到子坐标系的几何映射

        # 子窗口截取
        for i in range(len(labels)) :
            plot1.plot(x, data_sets[i], 
                       label        = labels[i], 
                       color        = colors[i], 
                       linestyle    = styles[i], 
                       linewidth    = 1.5)

        # 子窗口参数
        plot1.set_title('Noon Rush Hour', fontsize=25)      # 标题
        plot1.set_xlim(126, 162)
        # [10.5*12, 11.0*12, 11.5*12, 12.0*12, 12.5*12, 13.0*12, 13.5*12]
        plot1.set_xticks(np.arange(126, 168, 6))            # 七个刻度
        plot1.set_xticklabels(['10:30', '11:00', 
                               '11:30', '12:00', 
                               '12:30', '13:00', 
                               '13:30'], fontsize=8)        # 刻度标签
        plot1.set_ylim(200, 260)                            #### 显示范围 ####
        plot1.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, color='gray')

        # 箭头
        ax.annotate('', 
                    xy=(168, 140),       # 被注释点坐标
                    xytext=(144, 210),   # 起注释点坐标
                    arrowprops=dict(arrowstyle="->", color='red', linewidth=1.5))
    ################################################################
        plt.tight_layout()
        save_path = "fig_2_curves_04 " + f"{node+1}-02.{18+d}.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"✅ 图表已保存至: {save_path}")
    ################################################################

