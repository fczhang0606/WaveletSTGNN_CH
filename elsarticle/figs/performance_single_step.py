import numpy as np


def cal_RRSE(data1, data2) :

    tmp  = 0
    tmp_ = []

    for i in range(len(data1)) :
        tmp = 1000*((data2[i] - data1[i]) / data2[i])
        print("{:.4f}".format(tmp))
        tmp_.append(tmp)
    print("avg: {:.4f}".format(np.mean(tmp_)))

def cal_CORR(data1, data2) :

    tmp  = 0
    tmp_ = []

    for i in range(len(data1)) :
        tmp = 1000*((data1[i] - data2[i]) / data2[i])
        print("{:.4f}".format(tmp))
        tmp_.append(tmp)
    print("avg: {:.4f}".format(np.mean(tmp_)))


################################################################
traffic_RRSE_1 = [4183, 4350, 4394, 4434]
traffic_RRSE_2 = [4216, 4414, 4495, 4453]
traffic_CORR_1 = [9006, 8902, 8870, 8831]
traffic_CORR_2 = [8920, 8833, 8772, 8825]

cal_RRSE(traffic_RRSE_1, traffic_RRSE_2)  # 12.2657
cal_CORR(traffic_CORR_1, traffic_CORR_2)  # 7.3262
################################################################
# solar_RRSE_1 = [1629, 2190, 2982, 4264]
# solar_RRSE_2 = [1708, 2278, 2997, 4081]
# solar_CORR_1 = [9877, 9768, 9551, 9018]
# solar_CORR_2 = [9865, 9743, 9550, 9112]
solar_RRSE_1 = [1629, 2190, 2982]
solar_RRSE_2 = [1708, 2278, 2997]
solar_CORR_1 = [9877, 9768, 9551]
solar_CORR_2 = [9865, 9743, 9550]

# cal_RRSE(solar_RRSE_1, solar_RRSE_2)  # 11.2616
# cal_CORR(solar_CORR_1, solar_CORR_2)  # -1.6072
cal_RRSE(solar_RRSE_1, solar_RRSE_2)  # 29.9628
cal_CORR(solar_CORR_1, solar_CORR_2)  # 1.2957
################################################################
exchange_RRSE_1 = [172, 240, 333, 440]
exchange_RRSE_2 = [171, 240, 331, 436]
exchange_CORR_1 = [9819, 9735, 9580, 9399]
exchange_CORR_2 = [9798, 9724, 9601, 9418]

cal_RRSE(exchange_RRSE_1, exchange_RRSE_2)  # -5.2661
cal_CORR(exchange_CORR_1, exchange_CORR_2)  # -0.2325
################################################################
electricity_RRSE_1 = [726, 823, 896, 944]
electricity_RRSE_2 = [718, 844, 898, 962]
electricity_CORR_1 = [9450, 9321, 9202, 9164]
electricity_CORR_2 = [9494, 9387, 9321, 9279]

cal_RRSE(electricity_RRSE_1, electricity_RRSE_2)  # 8.6694
cal_CORR(electricity_CORR_1, electricity_CORR_2)  # -9.2065
################################################################

