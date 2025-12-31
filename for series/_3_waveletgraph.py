################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
import time
import warnings
warnings.filterwarnings('ignore')

from pytorch_wavelets import DWT1DForward, DWTForward
# git clone git@github.com:fbcotter/pytorch_wavelets.git
# cd PyTorchWavelets
# pip install -r requirements.txt
# python setup.py install
################################################################


################################################################
class WaveletGraph :


    def __init__(self, device, wave='db4', J=4) :

        self.device         = device
        self.wave           = wave
        self.J              = J

        self.dwt_1d         = DWT1DForward(J=J,         wave=wave, mode='zero').to(device)
        self.dwt_2d         = DWTForward  (J=min(J, 4), wave=wave, mode='zero').to(device)


    ### signal -- features
    def extract_features(self, x) :  # [B, 1, N, W]

        B, _, N, W  = x.shape
        features    = {}

        # 0-DWT                                                     # 0.0031
        x_1d        = x.reshape(B*N, 1, W)  # [B*N, 1, W]
        yl_1d, yh_1d= self.dwt_1d(x_1d)
        # len(yl_1d)=1      [B*N, 1, W_1]
        # len(yh_1d)=J      [B*N, 1, W_1], [B*N, 1, W_2]... [B*N, 1, W_J]

        # 1-wavelet_energy                                          # 0.0125
        wavelet_energy = []
        for j in range(self.J) :                                    # [B*N, 1, W_j] -- [B*N, 1]
            high_freq_energy = torch.mean(yh_1d[j]**2, dim=-1)
            wavelet_energy.append(high_freq_energy)
        low_freq_energy = torch.mean(yl_1d**2, dim=-1)              # [B*N, 1, W_0] -- [B*N, 1]
        wavelet_energy.append(low_freq_energy)

        wavelet_energy = torch.cat(wavelet_energy, dim=-1)          # [B*N, J+1]
        features['wavelet_energy'] = wavelet_energy.view(B, N, -1)  # [B, N, J+1]

        # 2-wavelet_entropy                                         # 0.0143
        wavelet_entropy = []
        for j in range(self.J) :
            energy_dist = F.softmax(yh_1d[j]**2, dim=-1)            # [B*N, 1, W_j]
            entropy     = -torch.sum(energy_dist*torch.log(energy_dist+1e-8), dim=-1)       # [B*N, 1]
            wavelet_entropy.append(entropy)
        features['wavelet_entropy'] = torch.cat(wavelet_entropy, dim=-1).view(B, N, -1)     # [B, N, J]

        # # 3-time_freq_map                                           # ?
        # if W >= 32 :                                                # lenth requirement
        #     time_freq_maps = self._create_time_freq_maps(x)
        #     features['time_freq_maps'] = time_freq_maps

        return features


    def _create_time_freq_maps(self, x) :  # [B, 1, N, W]

        B, _, N, W = x.shape

        # 
        time_freq_maps = []  # []:list  {}:dict/set
        for b in range(B) :
            batch_maps = []
            for n in range(N) :
                signal = x[b, 0, n, :]                      # [W]
                tf_map = self._signal_to_time_freq(signal)  # [F, W_tf]
                batch_maps.append(tf_map)
            batch_maps = torch.stack(batch_maps)            # [N, F, W_tf]，自然堆在头维？
            time_freq_maps.append(batch_maps)
        time_freq_maps = torch.stack(time_freq_maps)        # [B, N, F, W_tf]

        return time_freq_maps


    def _signal_to_time_freq(self, signal) :

        L           = len(signal)

        window_size = 64        # 数值讲究？
        stride      = 16

        if L < window_size :
            padded_signal   = F.pad(signal, (0, window_size - L))
            L               = window_size
        else :
            padded_signal   = signal


        num_windows = (L - window_size) // stride + 1
        spectrogram = []
        for i in range(num_windows) :
            start   = i * stride
            end     = start + window_size
            window  = padded_signal[start:end]

            fft         = torch.fft.fft(window)             # 核心操作
            magnitude   = torch.abs(fft[:window_size//2])   # 取正频率
            spectrogram.append(magnitude)


        if spectrogram :
            spectrogram = torch.stack(spectrogram).T        # [freq_bins, time_frames]
        else :
            # 如果无法创建频谱图，返回零矩阵
            spectrogram = torch.zeros(32, 32, device=self.device)

        return spectrogram


    ### features -- adjs
    def compute_similarity_matrix(self, features, similarity_types=None) :

        B, N, _ = features['wavelet_energy'].shape          # [B, N, J+1]

        if similarity_types is None :
            similarity_types = ['cosine', 
                                # 'euclidean', 
                                # 'pearson', 
                                # 'spearman', 
                                # 'wasserstein', 
                                # 'energy_correlation', 
                                # 'entropy_correlation', 
                                # 'time_freq_correlation']
            ]
        similarity_adjs = {}  # dict


        for type in similarity_types :
            adj = torch.zeros(B, N, N, device=self.device)

            for b in range(B) :
                if   type == 'cosine' :                     # 0.0000
                    adj[b] = self._cosine_similarity             (features['wavelet_energy'][b])
                elif type == 'euclidean' :                  # 0.6096
                    adj[b] = self._euclidean_similarity          (features['wavelet_energy'][b])
                elif type == 'pearson' :                    # 8.8426
                    adj[b] = self._pearson_similarity            (features['wavelet_energy'][b])
                elif type == 'spearman' :                   # 3.8462
                    adj[b] = self._spearman_similarity           (features['wavelet_energy'][b])
                elif type == 'wasserstein' :                # 0.4718
                    adj[b] = self._wasserstein_similarity        (features['wavelet_energy'][b])
                elif type == 'energy_correlation' :         # 0.0000
                    adj[b] = self._energy_correlation_similarity (features['wavelet_energy'][b])
                elif type == 'entropy_correlation' :        # 0.0000
                    adj[b] = self._entropy_correlation_similarity(features['wavelet_entropy'][b])
                elif type == 'time_freq_correlation' :      # ???
                    if 'time_freq_maps' in features :
                        adj[b] = self._time_freq_similarity      (features['time_freq_maps'][b])

            similarity_adjs[type] = adj  # [B, N, N]

        return similarity_adjs


    ### 
    def _cosine_similarity      (self, features) :  # [N, J+1]
        normalized = F.normalize(features, p=2, dim=1)          # 维度1的L2范数归一化
        return torch.mm(normalized, normalized.T)

    def _euclidean_similarity   (self, features) :  # [N, J+1]

        N   = features.shape[0]
        adj = torch.zeros(N, N, device=self.device)

        for i in range(N) :
            for j in range(N) :
                dist = torch.norm(features[i] - features[j])    # 不同节点在能量空间的欧氏距离
                adj[i, j] = 1.0 / (1.0 + dist)                  # 距离--相似度  # 转化粗糙

        return adj

    def _pearson_similarity     (self, features) :  # [N, J+1]

        features_np = features.cpu().numpy()
        N           = features_np.shape[0]
        adj         = torch.zeros(N, N, device=self.device)

        for i in range(N) :
            for j in range(N) :
                if i == j :
                    adj[i, j] = 1.0
                else :
                    corr, _ = pearsonr(features_np[i], features_np[j])  # 不同节点在能量空间的皮尔逊相关系数
                    adj[i, j] = (corr + 1) / 2                          # 相关系数--相似度  # 转化粗糙

        return adj

    def _spearman_similarity    (self, features) :  # [N, J+1]

        features_np = features.cpu().numpy()
        N           = features_np.shape[0]
        adj         = torch.zeros(N, N, device=self.device)

        for i in range(N) :
            for j in range(N) :
                if i == j :
                    adj[i, j] = 1.0
                else :
                    corr, _ = spearmanr(features_np[i], features_np[j])  # 不同节点在能量空间的斯皮尔曼相关系数
                    adj[i, j] = (corr + 1) / 2                           # 相关系数--相似度  # 转化粗糙

        return adj

    def _wasserstein_similarity (self, features) :  # [N, J+1]

        features_np = features.cpu().numpy()
        N           = features_np.shape[0]
        adj         = torch.zeros(N, N, device=self.device)

        for i in range(N) :
            for j in range(N) :
                if i == j :
                    adj[i, j] = 1.0
                else :
                    dist = wasserstein_distance(features_np[i], features_np[j])  # 不同节点在能量空间的瓦瑟斯坦距离
                    adj[i, j] = 1.0 / (1.0 + dist)                               # 距离--相似度  # 转化粗糙

        return adj


    ### 
    def _energy_correlation_similarity (self, features) :   # [N, J+1]
        energy_dist = F.softmax(features, dim=1)            # [N, J+1]
        return torch.mm(energy_dist, energy_dist.T)         # [N, N]

    def _entropy_correlation_similarity(self, features) :   # [N, J]
        normalized  = F.normalize(features, p=2, dim=1)     # [N, J]
        return torch.mm(normalized, normalized.T)

    def _time_freq_similarity(self, time_freq_maps) :       # [N, F, W]

        N   = time_freq_maps.shape[0]
        adj = torch.zeros(N, N, device=self.device)

        for i in range(N) :
            for j in range(N) :
                # 使用矩阵的Frobenius内积计算相似性
                map_i = time_freq_maps[i].flatten()
                map_j = time_freq_maps[j].flatten()
                # 余弦相似性
                sim = F.cosine_similarity(map_i.unsqueeze(0), map_j.unsqueeze(0), dim=1)
                adj[i, j] = sim.item()

        return adj


    ### wavelet graph analysis
    def wavelet_adjs(self, x) :                     # [B, 1, N, W]

        # energy, entropy, time_freq_maps
        features        = self.extract_features(x)

        # similarity_types = {
        #     cosine, euclidean, pearson, spearman, wasserstein, 
        #     energy_correlation, entropy_correlation, time_freq_correlation}
        wavelet_adjs    = self.compute_similarity_matrix(features)

        analyzed_adjs   = self._aggregate_similarity_analysis(wavelet_adjs)

        return features, wavelet_adjs, analyzed_adjs


    ### 
    def _aggregate_similarity_analysis(self, wavelet_adjs) :

        analyzed_adjs = {}

        for type, adj in wavelet_adjs.items() :  # x8 -- [B, N, N]

            B, N, _ = adj.shape     # [B, N, N]

            batch_analyzed_adjs = []# list
            for b in range(B) :

                sim = adj[b]        # [N, N]

                # 统计性质
                mask            = ~torch.eye(N, dtype=bool, device=self.device)
                mean_similarity = torch.mean(sim[mask])
                std_similarity  = torch.std (sim[mask])
                max_similarity  = torch.max (sim[mask])
                min_similarity  = torch.min (sim[mask])

                batch_analyzed_adjs.append({
                    'similarity_matrix':    sim.detach().cpu().numpy(), 
                    'mean_similarity':      mean_similarity.item(), 
                    'std_similarity':       std_similarity.item(), 
                    'max_similarity':       max_similarity.item(), 
                    'min_similarity':       min_similarity.item()
                })
            # 缺少一个对batch_analyzed_adjs的list综合处理
            analyzed_adjs[type] = batch_analyzed_adjs

        return analyzed_adjs

