'''
[cA_n, cD_n, cD_{n-1}, ..., cD_1] = 
pywt.wavedec(data, wavelet, mode='', level=None, axis=-1)

data:               [L, 1]              输入信号
wavelet:            ...                 各有特性
mode:               ''                  边界处理方式
level:              None                分解级别,None表示最大级别
axis:               -1                  计算轴
'''


import matplotlib.pyplot as plt
import numpy as np
import pywt


# 1. 创建测试信号
t = np.linspace(0, 1, 512, endpoint=False)
# 包含多个频率成分的信号
signal = (      np.sin(2 * np.pi * 2  * t) +    # 2Hz  低频
          0.8 * np.sin(2 * np.pi * 10 * t) +    # 10Hz 中频
          0.5 * np.sin(2 * np.pi * 25 * t) +    # 25Hz 中高频
          0.3 * np.sin(2 * np.pi * 40 * t))     # 40Hz 高频
signal[250:255] += 2  # 加入一个脉冲


# 2. 进行3级小波分解
wavelet = 'db4'
level   = 3
coeffs  = pywt.wavedec(signal, wavelet, level=level, mode='symmetric')


# 3. 显示分解结构信息
print("多级小波分解结构:")
print(f"原始信号长度: {len(signal)}")
print(f"分解级别: {level}")
print(f"系数列表长度: {len(coeffs)}")
for i, coef in enumerate(coeffs) :
    if i == 0 :
        print(f"cA{level} len = {len(coef)}")
    else :
        print(f"cD{level - i + 1} len = {len(coef)}")


# 4. 可视化多级分解结果
fig, axes = plt.subplots(len(coeffs)+1, 1, figsize=(12, 10))

# 绘制原始信号
axes[0].plot(t, signal)
axes[0].set_title('original signal')
axes[0].set_ylabel('amplitude')
axes[0].grid(True)

# 绘制各级系数
titles = [f'cA{level}', 
          f'cD{level}', 
          f'cD{level-1}', 
          f'cD1']

for i, (coef, title) in enumerate(zip(coeffs, titles)) :
    # 为每个系数创建对应的时间轴（由于下采样，长度减半）
    t_coef = np.linspace(t[0], t[-1], len(coef))
    axes[i+1].plot(t_coef, coef)
    axes[i+1].set_ylabel('amplitude')
    axes[i+1].set_title(f'{title} - len: {len(coef)}')
    axes[i+1].grid(True)

axes[-1].set_xlabel('time [s]')
plt.tight_layout()
plt.show()

