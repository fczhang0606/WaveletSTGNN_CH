####
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import math
import numpy as np


def histogram(title, categories, colors, values, ylim) :


    plt.figure(figsize=(20, 12))
    plt.gca().set_facecolor('#e0f0e0')      # 设置背景颜色
    plt.gcf().set_facecolor('#e0f0e0')      # 设置图像背景颜色


    plt.figtext(0.5, 0.03, title,                   # 图像标题
                ha='center', va='top', 
                fontsize=36, fontweight='bold', 
                color='black')
    plt.xticks([])                                  # X轴不显示刻度
    plt.ylabel('values', fontsize=30, labelpad=10)  # Y轴标签
    plt.ylim(ylim[0], ylim[1])                      # Y轴范围
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')


    # 柱状图
    bars = plt.bar(categories,                      # 类别
                   values,                          # 值
                   color=colors,                    # 颜色
                   edgecolor='black', linewidth=0.7)
    for bar in bars :
        height = bar.get_height()                   # 数值
        plt.text(bar.get_x() + bar.get_width()/2,   # x
                 (ylim[0] + height)/2,              # y
                 f'{height:.2f}',                   # 显示数值
                 ha='center', 
                 va='center', 
                 color='white', 
                 fontweight='bold', 
                 fontsize=30)


    # 图例
    legend_patches = [Patch(label=categories[i], color=colors[i]) 
                      for i in range(len(categories))]
    plt.legend(handles=legend_patches,              # 句柄
               loc='lower center', 
               bbox_to_anchor=(0.5, 0.95),          # 图像顶部中央
               ncol=len(categories),                # 水平排列所有图例项
               frameon=True, 
               facecolor='white', 
               edgecolor='black', 
               fontsize=24, 
               title_fontsize=30)

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(title, dpi=100, bbox_inches='tight')


if __name__ == "__main__" :


    # 模型 + 颜色
    categories  = ['DWISTGNN', 
                   'w/o DWT', 
                   'w/o DGC', 
                   'w/o GCN', 
                   'w/o FI']
    colors      = ['#8B4513', 
                   '#C0C0C0', 
                   '#87CEEB', 
                   '#9370DB', 
                   '#00008B']


    # 04-MAE
    values  = [18.01, 18.12, 18.39, 18.36, 18.13]
    ylim    = (17.8, 18.5)
    histogram('Ablation-PEMS04-MAE',    categories, colors, values, ylim)

    # 04-MAPE
    values  = [12.13, 12.17, 12.35, 12.29, 12.18]
    ylim    = (12, 12.4)
    histogram('Ablation-PEMS04-MAPE',   categories, colors, values, ylim)

    # 04-RMSE
    values  = [29.53, 29.69, 29.92, 29.94, 29.78]
    ylim    = (29.4, 30)
    histogram('Ablation-PEMS04-RMSE',   categories, colors, values, ylim)


    # 08-MAE
    values  = [13.21, 13.29, 13.43, 13.65, 13.28]
    ylim    = (13, 13.8)
    histogram('Ablation-PEMS08-MAE',    categories, colors, values, ylim)

    # 08-MAPE
    values  = [8.67, 8.75, 8.77, 8.94, 8.77]
    ylim    = (8.5, 9.0)
    histogram('Ablation-PEMS08-MAPE',   categories, colors, values, ylim)

    # 08-RMSE
    values  = [22.88, 23.08, 23.32, 23.46, 22.90]
    ylim    = (22.5, 23.6)
    histogram('Ablation-PEMS08-RMSE',   categories, colors, values, ylim)

