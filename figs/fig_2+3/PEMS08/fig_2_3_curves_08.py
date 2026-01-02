####
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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


    # PEMS08
    # July 1, 2016 to August 31, 2016 = 62 days = 62*288 = 17856
    # 2016.08.20
    # 0701～0819 = 50 days = 50*288 = 14400
    # 14400-14297 = 103
    p = 103         # 08.20     ################ 选择时间 ################
    l = 288         # 一日长度


     # searcher的结果
    rec = [(88, 9)]
    for node, d in rec :

        data_1_ = data_1[:, node, 0] # (17856,)
        data_2_ = data_2[:, node, 0] # (3567,)
        data_3_ = data_3[:, node, 0] # (3548,)
        data_4_ = data_4[:, node, 0] # (3548,)
        data_5_ = data_5[:, node, 0] # (3567,)
        data_6_ = data_6[:, node, 0] # (3567,)
        data_7_ = data_7[:, node, 0] # (3567,)
        data_8_ = data_8[:, node, 0] # (3567,)


        data_1__ = data_1_[14297+p+d*288:14297+p+d*288+l]     # 全局real的天内片段
        data_2__ = data_2_[19+p+d*288   :19+p+d*288+l]        # 测试集real的天内片段

        data_3__ = data_3_[p+d*288      :p+d*288+l]           # STWave-real的天内片段
        data_5__ = data_5_[19+p+d*288   :19+p+d*288+l]        # STIDGCN-real的天内片段
        data_7__ = data_7_[19+p+d*288   :19+p+d*288+l]        # DWISTGNN-real的天内片段

        data_4__ = data_4_[p+d*288      :p+d*288+l]           # STWave-pred的天内片段
        data_6__ = data_6_[19+p+d*288   :19+p+d*288+l]        # STIDGCN-pred的天内片段
        data_8__ = data_8_[19+p+d*288   :19+p+d*288+l]        # DWISTGNN-pred的天内片段


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


        ax.set_title('Model Comparison on PEMS08', fontsize=36)

        ax.set_xlabel(f'Node {node+1} in August {20+d}, 2016', 
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
                        width="50%", height="60%",               # 比例
                        loc='upper left', 
                        bbox_to_anchor=(0.32, -0.12, 0.5, 0.5),  # 位置+形变
                        bbox_transform=ax.transAxes)             # 从父坐标系到子坐标系的几何映射

        # 子窗口截取
        for i in range(len(labels)) :
            plot1.plot(x, data_sets[i], 
                       label        = labels[i], 
                       color        = colors[i], 
                       linestyle    = styles[i], 
                       linewidth    = 1.5)

        # 子窗口参数
        plot1.set_title('Morning Rush Hour', fontsize=18)   # 标题
        plot1.set_xlim(60, 108)
        # [5*12, 6*12, 7*12, 8*12, 9*12]
        plot1.set_xticks(np.arange(60, 120, 12))            # [5*12, 6*12, 7*12, 8*12, 9*12]    五个刻度
        plot1.set_xticklabels(['05:00', '06:00', 
                               '07:00', '08:00', 
                               '09:00'], fontsize=8)        # 刻度标签
        plot1.set_ylim(150, 410)                            #### 显示范围 ####
        plot1.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, color='gray')

        # 箭头
        ax.annotate('', 
                    xy=(128, 208),      # 被注释点坐标
                    xytext=(84, 280),   # 起注释点坐标
                    arrowprops=dict(arrowstyle="->", color='red', linewidth=1.5))
    ################################################################
        # 子窗口2
        plot2 = inset_axes(ax, 
                        width="50%", height="60%",               # 比例
                        loc='upper left', 
                        bbox_to_anchor=(0.65, -0.12, 0.5, 0.5),  # 位置+形变
                        bbox_transform=ax.transAxes)             # 从父坐标系到子坐标系的几何映射

        # 子窗口截取
        for i in range(len(labels)) :
            plot2.plot(x, data_sets[i], 
                       label        = labels[i], 
                       color        = colors[i], 
                       linestyle    = styles[i], 
                       linewidth    = 1.5)

        # 子窗口参数
        plot2.set_title('Evening Rush Hour', fontsize=18)   # 标题
        plot2.set_xlim(168, 216)
        # [14*12, 15*12, 16*12, 17*12, 18*12]
        plot2.set_xticks(np.arange(168, 228, 12))           # [14*12, 15*12, 16*12, 17*12, 18*12]   五个刻度
        plot2.set_xticklabels(['14:00', '15:00', 
                               '16:00', '17:00', 
                               '18:00'], fontsize=8)        # 刻度标签
        plot2.set_ylim(350, 480)                            #### 显示范围 ####
        plot2.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, color='gray')

        # 箭头
        ax.annotate('', 
                    xy=(224, 210),      # 被注释点坐标
                    xytext=(192, 350),  # 起注释点坐标
                    arrowprops=dict(arrowstyle="->", color='red', linewidth=1.5))
    ################################################################
        plt.tight_layout()
        save_path = "fig_2_curves_08 " + f"{node+1}-08.{20+d}.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"✅ 图表已保存至: {save_path}")
    ################################################################

