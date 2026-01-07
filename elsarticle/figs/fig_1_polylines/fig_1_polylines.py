####
import matplotlib.pyplot as plt
import numpy as np


def polyline(data, models, colors, title, horizons, y_lable, y_lim, y_ticks) :


    plt.figure(figsize=(20, 12), dpi=100)
    plt.title(title, fontsize=36)

    plt.xlabel("Horizon", fontsize=36)
    plt.xlim(min(horizons)-0.5, max(horizons)+0.5)
    plt.xticks(np.arange(min(horizons), max(horizons)+1, 1))

    plt.ylabel(y_lable, fontsize=36)
    plt.ylim(y_lim[0], y_lim[1])
    plt.yticks(y_ticks)

    plt.grid(True, linestyle='--', alpha=0.5)


    markers = ['o', 's', 'D']
    for i, model in enumerate(models) :
        plt.plot(horizons, 
                 data[model], 
                 color =colors[i], 
                 marker=markers[i], 
                 markersize=10, 
                 linewidth=2.5, 
                 label=model)
    plt.legend(loc='upper left', fontsize=30, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(title, dpi=100, bbox_inches='tight')


if __name__ == "__main__" :


    # 模型 + 颜色
    models = ['STWave', 'STIDGCN', 'DWISTGNN']
    colors = ["#157E1E", "#060cd5", "#ef0c0c"]

    # x
    horizons        = np.arange(1, 13)  # x轴点位

    # y
    title_04_MAE    = "Model Comparison on PeMS04 (MAE)"
    y_lable_04_MAE  = "MAE"
    y_lim_04_MAE    = (16, 20)
    y_ticks_04_MAE  = np.arange(16, 20, 0.1)
    data_04_MAE     = {
        'STWave':   [16.9913, 17.2709, 17.6240, 18.0658, 
                     18.1387, 18.4567, 18.5846, 18.8810, 
                     19.0064, 19.1906, 19.6526, 19.6933],  # 18.4630
        'STIDGCN':  [16.6109, 17.0452, 17.4157, 17.7158, 
                     17.9447, 18.1849, 18.4014, 18.5754, 
                     18.7606, 18.9451, 19.1487, 19.3612],  # 18.1758
        'DWISTGNN': [16.5036, 16.9242, 17.2939, 17.5680, 
                     17.8185, 18.0197, 18.2295, 18.3992, 
                     18.5698, 18.7334, 18.9217, 19.1569]   # 18.0115
                     }

    title_04_MAPE   = "Model Comparison on PeMS04 (MAPE)"
    y_lable_04_MAPE = "MAPE"
    y_lim_04_MAPE   = (0.110, 0.135)
    y_ticks_04_MAPE = np.arange(0.110, 0.135, 0.001)
    data_04_MAPE    = {
        'STWave':   [0.1156, 0.1163, 0.1181, 0.1212, 
                     0.1214, 0.1238, 0.1245, 0.1269, 
                     0.1274, 0.1288, 0.1330, 0.1325],  # 0.1241
        'STIDGCN':  [0.1128, 0.1159, 0.1186, 0.1195, 
                     0.1199, 0.1239, 0.1233, 0.1240, 
                     0.1272, 0.1266, 0.1280, 0.1323],  # 0.1227
        'DWISTGNN': [0.1114, 0.1147, 0.1163, 0.1184, 
                     0.1191, 0.1215, 0.1226, 0.1242, 
                     0.1250, 0.1264, 0.1269, 0.1286]   # 0.1213
                     }

    title_04_RMSE   = "Model Comparison on PeMS04 (RMSE)"
    y_lable_04_RMSE = "RMSE"
    y_lim_04_RMSE   = (26, 33)
    y_ticks_04_RMSE = np.arange(26, 33, 0.2)
    data_04_RMSE    = {
        'STWave':   [27.7196, 28.3613, 29.0164, 29.6748, 
                     29.9348, 30.3564, 30.6588, 31.0342, 
                     31.3071, 31.5739, 32.1540, 32.2921],  # 30.3403
        'STIDGCN':  [26.9039, 27.8247, 28.5030, 29.0430, 
                     29.4620, 29.8303, 30.1754, 30.4840, 
                     30.7720, 31.0479, 31.3384, 31.6287],  # 29.7511
        'DWISTGNN': [26.8142, 27.6708, 28.3418, 28.8464, 
                     29.2877, 29.6295, 29.9846, 30.2433, 
                     30.5169, 30.7404, 30.9923, 31.3135]   # 29.5318
                     }

    title_08_MAE    = "Model Comparison on PeMS08 (MAE)"
    y_lable_08_MAE  = "MAE"
    y_lim_08_MAE    = (11, 15)
    y_ticks_08_MAE  = np.arange(11, 15, 0.1)
    data_08_MAE     = {
        'STWave':   [11.8891, 12.2399, 12.5494, 12.8558, 
                     13.1180, 13.4033, 13.6201, 13.8353, 
                     13.9960, 14.1910, 14.4148, 14.7627],  # 13.4063
        'STIDGCN':  [11.7654, 12.1961, 12.5800, 12.9200, 
                     13.1814, 13.4229, 13.6818, 13.8856, 
                     14.0657, 14.2435, 14.4285, 14.7207],  # 13.4243
        'DWISTGNN': [11.6292, 12.0369, 12.3934, 12.7261, 
                     12.9736, 13.2296, 13.4223, 13.6458, 
                     13.8282, 13.9991, 14.1957, 14.4236]   # 13.2086
                     }

    title_08_MAPE   = "Model Comparison on PeMS08 (MAPE)"
    y_lable_08_MAPE = "MAPE"
    y_lim_08_MAPE   = (0.074, 0.100)
    y_ticks_08_MAPE = np.arange(0.074, 0.100, 0.002)
    data_08_MAPE    = {
        'STWave':   [0.0800, 0.0817, 0.0833, 0.0851, 
                     0.0869, 0.0897, 0.0908, 0.0918, 
                     0.0931, 0.0945, 0.0960, 0.0989],  # 0.0893
        'STIDGCN':  [0.0772, 0.0799, 0.0819, 0.0839, 
                     0.0857, 0.0874, 0.0889, 0.0905, 
                     0.0920, 0.0932, 0.0945, 0.0969],  # 0.0877
        'DWISTGNN': [0.0779, 0.0785, 0.0807, 0.0840, 
                     0.0843, 0.0873, 0.0874, 0.0890, 
                     0.0904, 0.0919, 0.0934, 0.0952]   # 0.0867
                     }

    title_08_RMSE   = "Model Comparison on PeMS08 (RMSE)"
    y_lable_08_RMSE = "RMSE"
    y_lim_08_RMSE   = (18, 27)
    y_ticks_08_RMSE = np.arange(18, 27, 0.2)
    data_08_RMSE    = {
        'STWave':   [19.8882, 20.8505, 21.6326, 22.3355, 
                     22.9192, 23.5174, 24.0071, 24.3219, 
                     24.7567, 25.1262, 25.4963, 25.9114],  # 23.3969
        'STIDGCN':  [19.4529, 20.5772, 21.4645, 22.3246, 
                     22.9196, 23.5257, 24.0644, 24.5803, 
                     24.9048, 25.2131, 25.4825, 25.8464],  # 23.3630
        'DWISTGNN': [19.2804, 20.3464, 21.1720, 21.8908, 
                     22.5090, 23.0196, 23.4920, 23.9177, 
                     24.2898, 24.5895, 24.8796, 25.2055]   # 22.8827
                     }


    # 绘制折线图
    polyline(data_04_MAE, models, colors, 
             title_04_MAE, horizons, 
             y_lable_04_MAE, y_lim_04_MAE, y_ticks_04_MAE)
    polyline(data_04_MAPE, models, colors, 
             title_04_MAPE, horizons, 
             y_lable_04_MAPE, y_lim_04_MAPE, y_ticks_04_MAPE)
    polyline(data_04_RMSE, models, colors, 
             title_04_RMSE, horizons, 
             y_lable_04_RMSE, y_lim_04_RMSE, y_ticks_04_RMSE)
    polyline(data_08_MAE, models, colors, 
             title_08_MAE, horizons, 
             y_lable_08_MAE, y_lim_08_MAE, y_ticks_08_MAE)
    polyline(data_08_MAPE, models, colors, 
             title_08_MAPE, horizons, 
             y_lable_08_MAPE, y_lim_08_MAPE, y_ticks_08_MAPE)
    polyline(data_08_RMSE, models, colors, 
             title_08_RMSE, horizons, 
             y_lable_08_RMSE, y_lim_08_RMSE, y_ticks_08_RMSE)

