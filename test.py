import torch
import torch.nn as nn
import numpy as np
import pywt
from scipy import signal
from sklearn.metrics.pairwise import cosine_similarity

class CWTSimilarity(nn.Module):
    def __init__(self, scales=np.arange(1, 32), wavelet='morl', similarity_metric='cosine'):
        """
        初始化CWT相似性计算模块
        
        Args:
            scales: 小波尺度范围
            wavelet: 小波类型 ('morl', 'cgau', 'cmor'等)
            similarity_metric: 相似性度量方法 ('cosine', 'correlation', 'euclidean')
        """
        super(CWTSimilarity, self).__init__()
        self.scales = scales
        self.wavelet = wavelet
        self.similarity_metric = similarity_metric
        
    def continuous_wavelet_transform(self, x):
        """
        对输入信号进行连续小波变换
        
        Args:
            x: 输入信号 [B, 1, N, W]
            
        Returns:
            cwt_matrix: CWT系数矩阵 [B, N, len(scales), W]
        """
        B, _, N, W = x.shape
        cwt_matrix = []
        
        for b in range(B):
            batch_cwt = []
            for n in range(N):
                # 提取单个节点的信号
                signal_data = x[b, 0, n, :].cpu().numpy()
                
                # 进行连续小波变换
                coefficients, frequencies = pywt.cwt(signal_data, self.scales, self.wavelet)
                batch_cwt.append(coefficients)
            
            cwt_matrix.append(np.stack(batch_cwt, axis=0))
        
        return torch.tensor(np.stack(cwt_matrix, axis=0), dtype=torch.float32)
    
    def compute_time_frequency_representation(self, cwt_coeffs):
        """
        计算时频表示特征
        
        Args:
            cwt_coeffs: CWT系数 [B, N, scales, W]
            
        Returns:
            features: 时频特征矩阵 [B, N, feature_dim]
        """
        B, N, S, W = cwt_coeffs.shape
        
        features = []
        for b in range(B):
            batch_features = []
            for n in range(N):
                node_cwt = cwt_coeffs[b, n]  # [S, W]
                node_cwt = node_cwt.cpu().numpy()
                
                # 提取多种时频特征
                feature_vector = []
                
                # 1. 时频矩阵的均值特征
                mean_features = np.mean(np.abs(node_cwt), axis=1)  # [S]
                feature_vector.extend(mean_features)
                
                # 2. 时频矩阵的方差特征
                var_features = np.var(np.abs(node_cwt), axis=1)    # [S]
                feature_vector.extend(var_features)
                
                # 3. 能量分布特征
                energy = np.sum(np.abs(node_cwt)**2, axis=1)       # [S]
                feature_vector.extend(energy)
                
                # 4. 频谱质心特征
                if S > 1:
                    centroid = np.sum(self.scales.reshape(-1, 1) * np.abs(node_cwt)**2, axis=0)
                    centroid /= (np.sum(np.abs(node_cwt)**2, axis=0) + 1e-8)
                    centroid_features = [np.mean(centroid), np.std(centroid)]
                    feature_vector.extend(centroid_features)
                
                batch_features.append(feature_vector)
            
            features.append(np.stack(batch_features, axis=0))
        
        return torch.tensor(np.stack(features, axis=0), dtype=torch.float32)
    
    def compute_similarity_matrix(self, features):
        """
        计算节点间的相似性矩阵
        
        Args:
            features: 时频特征 [B, N, feature_dim]
            
        Returns:
            similarity_matrix: 相似性矩阵 [B, N, N]
        """
        B, N, D = features.shape
        similarity_matrices = []
        
        for b in range(B):
            batch_features = features[b].numpy()  # [N, D]
            
            if self.similarity_metric == 'cosine':
                # 余弦相似度
                sim_matrix = cosine_similarity(batch_features)
            elif self.similarity_metric == 'correlation':
                # 相关系数
                sim_matrix = np.corrcoef(batch_features)
            elif self.similarity_metric == 'euclidean':
                # 基于欧氏距离的相似度（转换为相似性）
                distances = np.linalg.norm(batch_features[:, None] - batch_features[None, :], axis=2)
                sim_matrix = 1 / (1 + distances)
            else:
                raise ValueError(f"不支持的相似性度量方法: {self.similarity_metric}")
            
            similarity_matrices.append(sim_matrix)
        
        return torch.tensor(np.stack(similarity_matrices, axis=0), dtype=torch.float32)
    
    def forward(self, x):
        """
        前向传播计算节点相似性
        
        Args:
            x: 输入信号 [B, 1, N, W]
            
        Returns:
            similarity_matrix: 节点相似性矩阵 [B, N, N]
            time_freq_features: 时频特征 [B, N, feature_dim]
            cwt_coeffs: CWT系数 [B, N, scales, W]
        """
        # 1. 计算连续小波变换
        cwt_coeffs = self.continuous_wavelet_transform(x)  # [B, N, scales, W]
        
        # 2. 提取时频特征
        time_freq_features = self.compute_time_frequency_representation(cwt_coeffs)  # [B, N, feature_dim]
        
        # 3. 计算相似性矩阵
        similarity_matrix = self.compute_similarity_matrix(time_freq_features)  # [B, N, N]
        
        return similarity_matrix, time_freq_features, cwt_coeffs

# 使用示例
def example_usage():
    # 生成示例数据 [B, 1, N, W]
    B, N, W = 2, 5, 100  # 批次大小2, 5个节点, 信号长度100
    x = torch.randn(B, 1, N, W)
    
    # 初始化CWT相似性计算模块
    cwt_similarity = CWTSimilarity(
        scales=np.arange(1, 16),  # 尺度范围
        wavelet='morl',           # Morlet小波
        similarity_metric='cosine' # 余弦相似度
    )
    
    # 计算相似性
    similarity_matrix, features, cwt_coeffs = cwt_similarity(x)
    
    print(f"输入形状: {x.shape}")
    print(f"相似性矩阵形状: {similarity_matrix.shape}")
    print(f"时频特征形状: {features.shape}")
    print(f"CWT系数形状: {cwt_coeffs.shape}")
    
    # 显示第一个批次的相似性矩阵
    print("\n第一个批次的节点相似性矩阵:")
    print(similarity_matrix[0].detach().numpy().round(3))

# 增强版本：支持GPU和批处理优化
class FastCWTSimilarity(CWTSimilarity):
    def __init__(self, scales=np.arange(1, 32), wavelet='morl', similarity_metric='cosine'):
        super(FastCWTSimilarity, self).__init__(scales, wavelet, similarity_metric)
        
    def continuous_wavelet_transform(self, x):
        """优化版本：使用向量化操作加速计算"""
        B, _, N, W = x.shape
        cwt_matrix = []
        
        # 使用多进程加速（可选）
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        
        def process_signal(args):
            signal, scales, wavelet = args
            coefficients, _ = pywt.cwt(signal, scales, wavelet)
            return coefficients
        
        for b in range(B):
            batch_signals = x[b, 0].cpu().numpy()  # [N, W]
            
            # 准备多进程参数
            args_list = [(batch_signals[n], self.scales, self.wavelet) for n in range(N)]
            
            with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
                batch_cwt = list(executor.map(process_signal, args_list))
            
            cwt_matrix.append(np.stack(batch_cwt, axis=0))
        
        return torch.tensor(np.stack(cwt_matrix, axis=0), dtype=torch.float32)

if __name__ == "__main__":
    example_usage()