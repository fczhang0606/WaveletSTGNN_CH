'''
coefs, freqs = pywt.cwt(data, scales, wavelet, sampling_period=1.0, method='conv', axis=-1)

data:               [L, 1]              输入信号
scales:             [S, 1]              如何定值范围
wavelet:            ...                 各有特性
sampling_period:    float               采样周期,滑动;默认1.0,提供则返回的freqs为实际频率
method:             'conv' or 'fft'     积分计算方法
axis:               -1                  计算轴
coefs:              [S, L]              某个时刻某个频率下的强度
freqs:              [S, 1]              尺度对应频率 or 物理频率?
尺度对应频率:         f_a = (f_c) / (s * sampling_period)       f_c(小波中心频率)
'''


import matplotlib.pyplot as plt
import numpy as np
import pywt


# 1. 生成一个包含多个频率成分的测试信号
t = np.linspace(0, 1, 1000, endpoint=False)
# 信号由三部分组成：
# 0-0.4秒：     10Hz 正弦波
# 0.4-0.7秒：   25Hz 正弦波
# 0.7-1.0秒：   50Hz 正弦波 + 一个瞬时脉冲
signal = np.piecewise(t, [t < 0.4, (t >= 0.4) & (t < 0.7), t >= 0.7], 
[lambda t: np.sin(2 * np.pi * 10 * t), 
 lambda t: np.sin(2 * np.pi * 25 * t), 
 lambda t: np.sin(2 * np.pi * 50 * t) + (t == 0.75) * 5])  # 在0.75秒处加一个脉冲


# 2. 设置CWT参数
sampling_rate   = 1000                  # 采样率：1000 Hz
sampling_period = 1.0 / sampling_rate   # 采样周期：0.001秒

# 定义尺度范围。我们希望分析从1Hz到100Hz的频率。
# 尺度与频率成反比，所以最大频率对应最小尺度，最小频率对应最大尺度。
# 使用 pywt.frequency2scale 来辅助确定尺度范围（更精确的方法）
max_freq = 100
min_freq = 1
# 中心频率，对于 'cmor1.5-1.0' 大约是1.0
f_c = 1.0
# 计算对应的最大和最小尺度
max_scale = f_c / (min_freq * sampling_period)
min_scale = f_c / (max_freq * sampling_period)
# 创建尺度数组（这里用对数间隔更符合频率感知）
num_scales = 100
scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num_scales)

# 更简单直观的方法：直接定义一个较大的尺度范围，如1到128
# scales = np.arange(1, 128)


# 3. 执行连续小波变换
wavelet = 'cmor1.5-1.0'  # 使用复Morlet小波，带宽参数1.5，中心频率1.0
coefs, frequencies = pywt.cwt(signal, scales, wavelet, 
                              sampling_period=sampling_period, method='fft')


# 4. 可视化
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# 4.1 绘制原始信号
ax = axes[0]
ax.plot(t, signal)
ax.set_ylabel('Amplitude')
ax.set_title('Original Signal')
ax.grid(True)
# 标记不同频率区间
ax.axvline(x=0.4, color='r', linestyle='--', alpha=0.5)
ax.axvline(x=0.7, color='r', linestyle='--', alpha=0.5)
ax.text(0.2,  2, '10 Hz')
ax.text(0.55, 2, '25 Hz')
ax.text(0.85, 2, '50 Hz + Pulse')

# 4.2 绘制时频谱图（幅度谱）
ax = axes[1]
# 使用 pcolormesh 绘制，比 imshow 在坐标轴上更精确
im = ax.pcolormesh(t, frequencies, np.abs(coefs), shading='gouraud', cmap='jet')
ax.set_ylabel('Frequency [Hz]')
ax.set_title('Continuous Wavelet Transform (Magnitude)')
plt.colorbar(im, ax=ax, label='Magnitude')
ax.set_ylim(0, 100)  # 限制频率显示范围，以便看清细节


plt.tight_layout()
plt.show()

