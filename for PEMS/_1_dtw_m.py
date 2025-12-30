########################
####### 20250402 #######
########################


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pywt


def simple_attention_fusion(results_list) :

    L          = len(results_list)
    B, C, N, W = results_list[0].shape

    # 堆叠特征
    features = torch.stack(results_list, dim=1)  # [B, L, C, N, W]

    # 展平空间维度
    features_flat = features.reshape(B, L, C*N*W)

    # 计算注意力权重
    Q = features_flat
    K = features_flat
    V = features_flat

    # 缩放点积注意力
    attention_scores  = torch.matmul(Q, K.transpose(1, 2)) / (C*N*W)**0.5
    attention_weights = torch.softmax(attention_scores, dim=-1)  # [B, L, L]

    # 注意力输出
    attention_output  = torch.matmul(attention_weights, V)  # [B, L, C*N*W]

    # 取平均或加权
    fused_flat = attention_output.mean(dim=1)  # [B, C*N*W]

    # 恢复形状
    fused = fused_flat.reshape(B, C, N, W)

    return fused


def multi_wavelet_disentangle(x, wavelets, level) :

    # 输入
    # [B, C, N, W]
    # ["haar dmey db4 sym4 coif4 bior1.1 rbio1.1"]
    # level
    _, _, _, W  = x.shape   # B, C, N, W

    # 输出
    dtw_results_dict = {}
    xl_results_list  = []
    xh_results_list  = []


    for wavelet in wavelets :
        try :

            # 信号的离散小波分解
            # [B, C, N, W]  # 沿着最后一个维度进行小波分解
            coef = pywt.wavedec(x, wavelet, level=level, axis=-1)
            # [cA_3, cD_3, cD_2, cD_1]
            # [B, C, N, (l/2^3, l/2^3, l/2^2, l/2^1)]

            # 构建低频系数组
            coef_low  = [coef[0]]                           # 低位系数
            for i in range(len(coef) - 1) :
                coef_low.append(np.zeros_like(coef[1+i]))   # 高位补零
            # [cA_3, 0, 0, 0]
            # [B, C, N, (l/2^3, l/2^3, l/2^2, l/2^1)]

            # 构建高频系数组
            coef_high = [np.zeros_like(coef[0])]            # 低位补零
            for i in range(len(coef) - 1) :
                coef_high.append(coef[1+i])                 # 高位系数
            # [0, cD_3, cD_2, cD_1]
            # [B, C, N, (l/2^3, l/2^3, l/2^2, l/2^1)]

            # 两组分别重构：低频分量、高频分量
            xl = pywt.waverec(coef_low,  wavelet, axis=-1)  # [cA_3, 0, 0, 0]       -- [B, C, N, W]
            xh = pywt.waverec(coef_high, wavelet, axis=-1)  # [0, cD_3, cD_2, cD_1] -- [B, C, N, W]

            # 边界效应：信号的起点和终点附近，因滤波点缺失，变换结果出现失真
            # 确保重构的信号长度一致
            if xl.shape[-1] != W :
                xl = xl[..., :W]  # :w表示在最后一个维度上只取前w个元素。不足w则不变？
            if xh.shape[-1] != W :
                xh = xh[..., :W]

            # 保存结果
            dtw_results_dict[wavelet] = {'xl': xl, 'xh': xh}
            xl_results_list.append(xl)
            xh_results_list.append(xh)

        except Exception as e :
            print(f"小波 {wavelet} 变换失败: {e}")
            continue

    # 检查是否有成功的小波变换
    if len(xl_results_list) == 0 :
        raise ValueError("所有小波变换都失败了")

    # 融合: [B, C, N, W]
    xl_fused = np.mean(np.stack(xl_results_list, axis=0), axis=0)  # 低频平均
    xh_fused = np.mean(np.stack(xh_results_list, axis=0), axis=0)  # 高频平均
    # xh_fused = np.max (np.stack(xh_results_list, axis=0), axis=0)  # 高频取大

    # 自适应融合
    #xl_fused = simple_attention_fusion([torch.tensor(xl_res) for xl_res in xl_results_list]).numpy()
    #xh_fused = simple_attention_fusion([torch.tensor(xh_res) for xh_res in xh_results_list]).numpy()

    return dtw_results_dict, xl_fused, xh_fused

