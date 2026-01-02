import numpy as np


def cal(data1, data2) :

    tmp  = 0
    tmp_ = []

    for i in range(len(data1)) :
        tmp = 1000*((data2[i] - data1[i]) / data2[i])
        print("{:.4f}".format(tmp))
        tmp_.append(tmp)
    print("avg: {:.4f}".format(np.mean(tmp_)))


################################################################
MAE_1  = [18.01, 18.91, 13.21]
MAE_2  = [18.16, 19.26, 13.42]
RMSE_1 = [29.53, 32.39, 22.88]
RMSE_2 = [29.77, 32.51, 23.28]
MAPE_1 = [12.13, 7.88, 8.67]
MAPE_2 = [12.24, 8.11, 8.77]


cal(MAE_1,  MAE_2)   # 14.0269
cal(RMSE_1, RMSE_2)  # 9.6450
cal(MAPE_1, MAPE_2)  # 16.2498
################################################################

