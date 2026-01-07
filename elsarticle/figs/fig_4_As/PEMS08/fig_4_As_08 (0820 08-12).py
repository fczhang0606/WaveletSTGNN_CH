####
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import time


def As_heatmap(As, p, lp, n, ln) :


    # 数据预处理：时间范围、空间范围
    As_ = As[p:p+lp, n:n+ln, n:n+ln]

    for i in range(ln) :            # 节点遍历
        As_[:, i, i] = 0            # 对角线元素置为0
    As_mean = np.mean(As_, axis=0)  # 时长平均值
    As_mean = (As_mean - np.min(As_mean)) / \
              (np.max(As_mean) - np.min(As_mean))  # 归一化


    # 绘制热力图
    plt.figure(figsize=(20, 12), facecolor='white')         # 画布
    plt.title (f'PEMS08 2016.08.{20+(p-103)//288} {(p-103)%288//12}:00', 
               fontsize=30, pad=20)                         # 标题
    plt.xlabel('Nodes-X', fontsize=30)                      # X
    plt.xticks(np.arange(0, ln, 1), np.arange(0, ln, 1))    # [0, 1, 2, ..., ln-1]
    plt.ylabel('Nodes-Y', fontsize=30)                      # Y
    plt.yticks(np.arange(0, ln, 1), np.arange(0, ln, 1))    # [0, 1, 2, ..., ln-1]
    plt.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.7)


    colors  = ["#E0E0E0", "#EE0A0A"]  # 浅灰色 -> 深红色
    cmap    = LinearSegmentedColormap.from_list("custom_cmap", colors)
    im = plt.imshow(As_mean, 
                    cmap    = cmap,     # 自定义
                    origin  = 'lower',  # 原点在左下角
                    aspect  = 'equal')  # 保持网格正方形
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Adjacency Weights', rotation=270, labelpad=15, fontsize=20)


    plt.tight_layout()
    plt.savefig("fig_4_As_08 " + 
                f"08.{20+(p-103)//288} {(p-103)%288//12}:00" + 
                ".png", 
                dpi=100, bbox_inches='tight')


def As_heatmap_D(As, p1, lp1, p2, lp2, n, ln) :


    # 数据预处理：时间范围、空间范围
    As_1 = As[p1:p1+lp1, n:n+ln, n:n+ln]
    As_2 = As[p2:p2+lp2, n:n+ln, n:n+ln]

    for i in range(ln) :                # 节点遍历
        As_1[:, i, i] = 0               # 对角线元素置为0
        As_2[:, i, i] = 0               # 对角线元素置为0
    As_1_mean = np.mean(As_1, axis=0)   # 时长平均值
    As_2_mean = np.mean(As_2, axis=0)   # 时长平均值

    # 归一化值
    As_1_mean   = (As_1_mean - np.min(As_1_mean)) / \
                    (np.max(As_1_mean) - np.min(As_1_mean))  # 归一化
    As_2_mean   = (As_2_mean - np.min(As_2_mean)) / \
                    (np.max(As_2_mean) - np.min(As_2_mean))  # 归一化
    # 差值
    As_mean     = As_2_mean - As_1_mean


    # 绘制热力图
    plt.figure(figsize=(20, 12), facecolor='white')         # 画布
    plt.title (f"PEMS08 2016.08.{20+(p1-103)//288} [{(p1-103)%288//12}:00-{(p2-103)%288//12}:00]", 
               fontsize=30, pad=20)                         # 标题
    plt.xlabel('Nodes-X', fontsize=30)                      # X
    plt.xticks(np.arange(0, ln, 1), np.arange(0, ln, 1))    # [0, 1, 2, ..., ln-1]
    plt.ylabel('Nodes-Y', fontsize=30)                      # Y
    plt.yticks(np.arange(0, ln, 1), np.arange(0, ln, 1))    # [0, 1, 2, ..., ln-1]
    plt.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.7)


    colors  = ["#E0E0E0", "#EE0A0A"]  # 浅灰色 -> 深红色
    cmap    = LinearSegmentedColormap.from_list("custom_cmap", colors)
    im = plt.imshow(As_mean, 
                    cmap    = cmap,     # 自定义
                    origin  = 'lower',  # 原点在左下角
                    aspect  = 'equal',  # 保持网格正方形
                    vmin=0, vmax=0.5)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Adjacency Weights', rotation=270, labelpad=15)


    plt.tight_layout()
    plt.savefig("fig_4_As_08 " + 
                f"08.{20+(p1-103)//288} [{(p1-103)%288//12}:00-{(p2-103)%288//12}:00]" + 
                ".png", 
                dpi=100, bbox_inches='tight')


if __name__ == "__main__" :


    # 载入文件
    file    = './PEMS08_As.npy' # 邻接矩阵
    As      = np.load(file)     # [3567, 170, 170]
    flag    = np.any(As<0)      # 检查是否有负值
    print("As<0: ", flag)       # False


    # PEMS08
    # July 1, 2016 to August 31, 2016 = 62 days = 62*288 = 17856

    # 2016.08.20
    # 0701～0819 = 50 days = 50*288 = 14400
    # 14400-14297 = 103
    p0  = 103       # 08.20     ################ 选择时间 ################
    d   = 0         # (17856-14400)/288=12, [0, 11], 17856取不到所以实际上取不到11
    h   = 0         # [0, 23]
    ld  = 288       # 一日长度
    lh  = 12        # 一时长度

    # 08.20-08:00
    d   = 0         # 08.20
    h   = 8         # 08:00
    p   = p0+d*ld+h*lh
    lp  = 3         # 
    n   = 0         # nodes
    ln  = 60        # [n, n+ln-1]
    As_heatmap(As, p, lp, n, ln)  # 绘制热力图
    time.sleep(1)

    # 08.20-12:00
    d   = 0         # 08.20
    h   = 12        # 12:00
    p   = p0+d*ld+h*lh
    lp  = 3         # 
    n   = 0         # nodes
    ln  = 60        # [n, n+ln-1]
    As_heatmap(As, p, lp, n, ln)  # 绘制热力图
    time.sleep(1)


    # 绘制差分热力图
    for i in range(1, 5) :  # 4小时段
        d   = 0             # 08.20
        h   = 8 + i         # 08:00-12:00

        p1  = p0+d*ld+8*lh  # 08.20-08:00
        lp1 = 3             # 
        p2  = p0+d*ld+h*lh  # 08.20-
        lp2 = 3             # 
        n   = 0             # nodes
        ln  = 60            # [n, n+ln-1]
        As_heatmap_D(As, p1, lp1, p2, lp2, n, ln)  # 绘制差分热力图
        time.sleep(1)

