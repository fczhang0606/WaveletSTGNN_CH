import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pywt
from matplotlib import cm
from scipy import signal
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean, correlation
import warnings
warnings.filterwarnings('ignore')

# 设置参数
fs = 1000  # 采样频率
t = np.linspace(0, 1, fs)  # 时间向量

# 生成两个测试信号（长度相等）
# 信号1：频率变化的信号
signal1 = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)
signal1[t > 0.5] += np.sin(2 * np.pi * 100 * t[t > 0.5])

# 信号2：包含脉冲和频率变化的信号（与信号1有部分相似性）
signal2 = np.sin(2 * np.pi * 10 * t) + 0.3 * np.sin(2 * np.pi * 50 * t)  # 相似频率成分
signal2[t > 0.3] += np.sin(2 * np.pi * 80 * t[t > 0.3])  # 不同频率
signal2[500:520] += 3  # 添加脉冲

# 确保信号长度相等
assert len(signal1) == len(signal2), "信号长度必须相等"

# 连续小波变换参数
scales = np.arange(1, 128)  # 尺度范围
wavelet = 'cmor1.5-1.0'     # 复Morlet小波

# 对两个信号进行连续小波变换
coef1, freqs1 = pywt.cwt(signal1, scales, wavelet, sampling_period=1/fs)
coef2, freqs2 = pywt.cwt(signal2, scales, wavelet, sampling_period=1/fs)

# 转换为幅度（取绝对值）
magnitude1 = np.abs(coef1)
magnitude2 = np.abs(coef2)

# 创建时间网格和频率网格
T, F = np.meshgrid(t, freqs1)

# ============================================================================
# 时频分布相似性度量函数
# ============================================================================

def calculate_similarity_metrics(tf1, tf2):
    """计算两个时频分布之间的多种相似性度量"""
    
    # 将矩阵展平为一维向量
    vec1 = tf1.flatten()
    vec2 = tf2.flatten()
    
    metrics = {}
    
    # 1. 余弦相似度
    metrics['cosine_similarity'] = cosine_similarity([vec1], [vec2])[0][0]
    
    # 2. 欧几里得距离（转换为相似度：1/(1+距离)）
    euclidean_dist = euclidean(vec1, vec2)
    metrics['euclidean_similarity'] = 1 / (1 + euclidean_dist)
    
    # 3. 皮尔逊相关系数
    correlation_coef = np.corrcoef(vec1, vec2)[0, 1]
    metrics['pearson_correlation'] = correlation_coef if not np.isnan(correlation_coef) else 0
    
    # 4. 互相关最大值（时频分布的时间对齐相似性）
    cross_corr = signal.correlate2d(tf1, tf2, mode='same')
    metrics['max_cross_correlation'] = np.max(cross_corr) / np.sqrt(np.sum(tf1**2) * np.sum(tf2**2))
    
    # 5. 结构相似性指数 (SSIM)
    # 由于SSIM需要图像块，我们计算平均SSIM
    from skimage.metrics import structural_similarity as ssim
    try:
        # 调整图像大小以适应SSIM计算
        min_dim = min(tf1.shape[0], tf1.shape[1], 64)
        tf1_resized = signal.resample(tf1, min_dim, axis=0)
        tf1_resized = signal.resample(tf1_resized, min_dim, axis=1)
        tf2_resized = signal.resample(tf2, min_dim, axis=0)
        tf2_resized = signal.resample(tf2_resized, min_dim, axis=1)
        
        metrics['ssim'] = ssim(tf1_resized, tf2_resized, data_range=tf1_resized.max()-tf1_resized.min())
    except:
        metrics['ssim'] = 0
    
    # 6. 能量分布相似度（基于能量分布的KL散度）
    energy1 = tf1 / np.sum(tf1)
    energy2 = tf2 / np.sum(tf2)
    
    # 避免log(0)的情况
    epsilon = 1e-10
    energy1 = energy1 + epsilon
    energy2 = energy2 + epsilon
    energy1 = energy1 / np.sum(energy1)
    energy2 = energy2 / np.sum(energy2)
    
    # KL散度（转换为相似度）
    kl_divergence = np.sum(energy1 * np.log(energy1 / energy2))
    metrics['energy_similarity'] = 1 / (1 + abs(kl_divergence))
    
    # 7. 频率成分相似度（基于主要频率成分）
    freq_profile1 = np.mean(tf1, axis=1)  # 时间平均的频率分布
    freq_profile2 = np.mean(tf2, axis=1)
    metrics['frequency_profile_similarity'] = cosine_similarity([freq_profile1], [freq_profile2])[0][0]
    
    # 8. 时间轮廓相似度（基于频率平均的时间分布）
    time_profile1 = np.mean(tf1, axis=0)  # 频率平均的时间分布
    time_profile2 = np.mean(tf2, axis=0)
    metrics['time_profile_similarity'] = cosine_similarity([time_profile1], [time_profile2])[0][0]
    
    return metrics

def calculate_comprehensive_similarity_score(metrics):
    """计算综合相似度得分（加权平均）"""
    weights = {
        'cosine_similarity': 0.2,
        'pearson_correlation': 0.2,
        'max_cross_correlation': 0.15,
        'ssim': 0.15,
        'energy_similarity': 0.1,
        'frequency_profile_similarity': 0.1,
        'time_profile_similarity': 0.1
    }
    
    weighted_sum = 0
    total_weight = 0
    
    for metric_name, weight in weights.items():
        if metric_name in metrics:
            # 将相似度归一化到0-1范围
            normalized_score = (metrics[metric_name] + 1) / 2 if metric_name in ['pearson_correlation'] else metrics[metric_name]
            weighted_sum += normalized_score * weight
            total_weight += weight
    
    return weighted_sum / total_weight if total_weight > 0 else 0

# 计算相似性度量
similarity_metrics = calculate_similarity_metrics(magnitude1, magnitude2)
comprehensive_score = calculate_comprehensive_similarity_score(similarity_metrics)

# ============================================================================
# 可视化结果
# ============================================================================

# 绘制三维时频分布图
fig = plt.figure(figsize=(20, 12))

# 第一个信号的三维时频图
ax1 = fig.add_subplot(231, projection='3d')
surf1 = ax1.plot_surface(T, F, magnitude1, cmap='viridis', 
                        linewidth=0, antialiased=True, alpha=0.8)
ax1.set_title('信号1的三维时频分布', fontsize=12, fontname='SimHei')
ax1.set_xlabel('时间 (s)')
ax1.set_ylabel('频率 (Hz)')
ax1.set_zlabel('幅度')
ax1.view_init(30, 45)

# 第二个信号的三维时频图
ax2 = fig.add_subplot(232, projection='3d')
surf2 = ax2.plot_surface(T, F, magnitude2, cmap='plasma', 
                        linewidth=0, antialiased=True, alpha=0.8)
ax2.set_title('信号2的三维时频分布', fontsize=12, fontname='SimHei')
ax2.set_xlabel('时间 (s)')
ax2.set_ylabel('频率 (Hz)')
ax2.set_zlabel('幅度')
ax2.view_init(30, 45)

# 两个信号的时频分布差异
difference = np.abs(magnitude1 - magnitude2)
ax3 = fig.add_subplot(233, projection='3d')
surf3 = ax3.plot_surface(T, F, difference, cmap='hot', 
                        linewidth=0, antialiased=True, alpha=0.8)
ax3.set_title('时频分布差异', fontsize=12, fontname='SimHei')
ax3.set_xlabel('时间 (s)')
ax3.set_ylabel('频率 (Hz)')
ax3.set_zlabel('差异幅度')
ax3.view_init(30, 45)

# 二维时频图对比
ax4 = fig.add_subplot(234)
im1 = ax4.imshow(magnitude1, extent=[t.min(), t.max(), freqs1.min(), freqs1.max()], 
                aspect='auto', cmap='viridis', origin='lower')
ax4.set_title('信号1 - 时频图')
ax4.set_xlabel('时间 (s)')
ax4.set_ylabel('频率 (Hz)')
plt.colorbar(im1, ax=ax4)

ax5 = fig.add_subplot(235)
im2 = ax5.imshow(magnitude2, extent=[t.min(), t.max(), freqs2.min(), freqs2.max()], 
                aspect='auto', cmap='plasma', origin='lower')
ax5.set_title('信号2 - 时频图')
ax5.set_xlabel('时间 (s)')
ax5.set_ylabel('频率 (Hz)')
plt.colorbar(im2, ax=ax5)

ax6 = fig.add_subplot(236)
im3 = ax6.imshow(difference, extent=[t.min(), t.max(), freqs1.min(), freqs1.max()], 
                aspect='auto', cmap='hot', origin='lower')
ax6.set_title('时频分布差异图')
ax6.set_xlabel('时间 (s)')
ax6.set_ylabel('频率 (Hz)')
plt.colorbar(im3, ax=ax6)

plt.tight_layout()
plt.show()

# ============================================================================
# 打印相似性分析结果
# ============================================================================

print("=" * 60)
print("时频分布相似性分析结果")
print("=" * 60)

print(f"\n综合相似度得分: {comprehensive_score:.4f}")
print(f"相似度等级: {'极高' if comprehensive_score > 0.8 else '高' if comprehensive_score > 0.6 else '中等' if comprehensive_score > 0.4 else '低' if comprehensive_score > 0.2 else '极低'}")

print("\n详细相似性度量:")
print("-" * 40)
for metric_name, value in similarity_metrics.items():
    print(f"{metric_name:30s}: {value:.4f}")

print("\n信号基本信息:")
print("-" * 40)
print(f"信号长度: {len(signal1)} 个采样点")
print(f"采样频率: {fs} Hz")
print(f"分析时间: {t.min():.2f} - {t.max():.2f} s")
print(f"频率范围: {freqs1.min():.2f} - {freqs1.max():.2f} Hz")

# ============================================================================
# 绘制相似性分析图表
# ============================================================================

# 创建相似性度量雷达图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 雷达图
metrics_names = ['cosine_similarity', 'pearson_correlation', 'max_cross_correlation', 
                 'ssim', 'energy_similarity', 'frequency_profile_similarity', 'time_profile_similarity']
metrics_values = [similarity_metrics[name] for name in metrics_names]

# 归一化到0-1范围（对于可能为负值的指标）
normalized_values = [(val + 1) / 2 if name in ['pearson_correlation'] else val for name, val in zip(metrics_names, metrics_values)]

angles = np.linspace(0, 2*np.pi, len(metrics_names), endpoint=False).tolist()
angles += angles[:1]  # 闭合雷达图
normalized_values += normalized_values[:1]

ax1.plot(angles, normalized_values, 'o-', linewidth=2, label='相似度')
ax1.fill(angles, normalized_values, alpha=0.25)
ax1.set_xticks(angles[:-1])
ax1.set_xticklabels([name.replace('_', '\n') for name in metrics_names])
ax1.set_ylim(0, 1)
ax1.set_title('时频分布相似性雷达图', fontsize=14, fontname='SimHei')
ax1.grid(True)

# 条形图
ax2.bar(range(len(metrics_names)), normalized_values, color=plt.cm.viridis(np.linspace(0, 1, len(metrics_names))))
ax2.set_xticks(range(len(metrics_names)))
ax2.set_xticklabels([name.replace('_', '\n') for name in metrics_names], rotation=45)
ax2.set_ylabel('相似度 (0-1)')
ax2.set_title('各度量指标相似度对比', fontsize=14, fontname='SimHei')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# 频率剖面和时间剖面对比
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 频率剖面对比
freq_profile1 = np.mean(magnitude1, axis=1)
freq_profile2 = np.mean(magnitude2, axis=1)
ax1.plot(freqs1, freq_profile1, label='信号1', linewidth=2)
ax1.plot(freqs1, freq_profile2, label='信号2', linewidth=2)
ax1.set_xlabel('频率 (Hz)')
ax1.set_ylabel('平均幅度')
ax1.set_title('频率剖面对比')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 时间剖面对比
time_profile1 = np.mean(magnitude1, axis=0)
time_profile2 = np.mean(magnitude2, axis=0)
ax2.plot(t, time_profile1, label='信号1', linewidth=2)
ax2.plot(t, time_profile2, label='信号2', linewidth=2)
ax2.set_xlabel('时间 (s)')
ax2.set_ylabel('平均幅度')
ax2.set_title('时间剖面对比')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

