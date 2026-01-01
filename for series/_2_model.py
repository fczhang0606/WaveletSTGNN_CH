########################
####### 20250402 #######
########################



# 
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from _1_dtw_s import *
from _1_dtw_m import *
from _3_graph import *
from _3_layer import *



###### 门控线性单元： Gated Linear Units ######
class GLU(nn.Module) :


    def __init__(self, channels, dropout) :

        super(GLU, self).__init__()

        self.conv1   = nn.Conv2d(channels, channels, (1, 1))
        self.conv2   = nn.Conv2d(channels, channels, (1, 1))
        self.conv3   = nn.Conv2d(channels, channels, (1, 1))
        self.dropout = nn.Dropout(dropout)


    # x，两路变形，再汇聚
    def forward(self, x) :

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)

        return out



###### FFD: 前馈神经网络 ######
class FeedForward(nn.Module) :


    def __init__(self, ftr_Mtx, res_en=False) :

        super(FeedForward, self).__init__()

        self.res_en = res_en
        self.L      = len(ftr_Mtx) - 1  # 层数
        self.linear = nn.ModuleList([nn.Linear(ftr_Mtx[i], ftr_Mtx[i+1]) for i in range(self.L)])
        self.ln     = nn.LayerNorm(ftr_Mtx[self.L], elementwise_affine=False)


    def forward(self, inputs) :

        x = inputs

        for i in range(self.L) :
            x = self.linear[i](x)
            if i != self.L-1 :
                x = F.relu(x)
            # else :  # 最后一层
                # x = self.ln(x)

        if self.res_en :  # 残差
            x += inputs
            x = self.ln(x)

        return x



###### 注意力机制： https://zhuanlan.zhihu.com/p/414084879 ######
class Adaptive_Fusion(nn.Module) :


    def __init__(self, heads, dims) :

        super(Adaptive_Fusion, self).__init__()

        features = heads * dims  # 2C = h * d
        self.h   = heads
        self.d   = dims

        self.Q = FeedForward([features, features])
        self.K = FeedForward([features, features])
        self.V = FeedForward([features, features])
        self.O = FeedForward([features, features])

        self.ln= nn.LayerNorm(features, elementwise_affine=False)
        self.ff= FeedForward([features, features, features], True)


    def forward(self, xl, xh, Mask=False) :  # Mask=True ！！！
        # [B, 2C, N, W] + [B, 2C, N, W]

        xl = xl.transpose(1, 3)  # -- [B, W, N, 2C]
        xh = xh.transpose(1, 3)  # -- [B, W, N, 2C]

        # Q要什么，K有什么，QK匹配吸收V
        queryl = self.Q(xl)              # -- [B, W, N, 2C]
        keyh   = torch.relu(self.K(xh))  # -- [B, W, N, 2C]
        valueh = torch.relu(self.V(xh))  # -- [B, W, N, 2C]

        queryl = torch.cat(torch.split(queryl, self.d, -1), 0).permute(0, 2, 1, 3)  # [h*B, N, W, d]
        keyh   = torch.cat(torch.split(keyh,   self.d, -1), 0).permute(0, 2, 3, 1)  # [h*B, N, d, W]
        valueh = torch.cat(torch.split(valueh, self.d, -1), 0).permute(0, 2, 1, 3)  # [h*B, N, W, d]

        attentionh = torch.matmul(queryl, keyh)  # [h*B, N, W, W]

        if Mask :
            batch_size  = xl.shape[0]  # B
            windows     = xl.shape[1]  # W
            nodes       = xl.shape[2]  # N
            mask = torch.ones(windows, windows).to(xl.device)  # [W, W]
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)  # [1, 1, W, W]
            mask = mask.repeat(self.h*batch_size, nodes, 1, 1)  # [h*B, N, W, W]
            mask = mask.to(torch.bool)
            zero_vec = (-2**15 + 1)*torch.ones_like(attentionh).to(xl.device)  # [h*B, N, W, W]
            attentionh = torch.where(mask, attentionh, zero_vec)

        attentionh /= (self.d ** 0.5)           # scaled
        attentionh = F.softmax(attentionh, -1)  # [h*B, N, W, W]

        value = torch.matmul(attentionh, valueh)     # [h*B, N, W, d]
        value = torch.cat(torch.split(value, value.shape[0]//self.h, 0), -1)\
                               .permute(0, 2, 1, 3)  # [B, W, N, 2C]
        value = self.O(value)  # xl吸收xh的部分

        value_fse = xl + value
        value_fse = self.ln(value_fse)
        value_fse = self.ff(value_fse).transpose(1, 3)  # -- [B, 2C, N, W]

        return value_fse



###### 扩散卷积 ######
class Diffusion_GCN(nn.Module) :


    def __init__(self, channels, diffusion_k, dropout) :

        super().__init__()

        self.diffusion_k    = diffusion_k
        self.conv           = nn.Conv2d(diffusion_k*channels, channels, (1, 1))  # 每k的结果concat后卷积
        self.dropout        = nn.Dropout(dropout)


    def forward(self, x, adj) :

        out = []  # 保存每层卷积的结果

        for i in range(0, self.diffusion_k) :

            if adj.dim() == 3 :
                # [B, 2C, N, W] + [B, N, N] -- [B, 2C, N, W]
                x = torch.einsum('bcnt, bnm->bcmt', x, adj).contiguous()  # torch.einsum详解
                out.append(x)
            elif adj.dim() == 2 :
                x = torch.einsum('bcnt, nm ->bcmt', x, adj).contiguous()
                out.append(x)

        x = torch.cat(out, dim=1)  # k-mix-hop
        x = self.conv(x)
        output = self.dropout(x)   # dropout

        return output



###### 动态空间卷积 ######
class DGCN(nn.Module) :


    def __init__(self, channels, diffusion_k, dropout, emb=None) :

        super().__init__()

        self.conv = nn.Conv2d(channels, channels, (1, 1))

        ### 原构图
        # self.GC = Graph_Generator(nodes, channels, diffusion_k, dropout)

        self.gcn  = Diffusion_GCN(channels, diffusion_k, dropout)

        self.emb  = emb


    # 
    def forward(self, x, adj) :  # [B, 2C, N, W] + [B, N, N] -- [B, 2C, N, W]

        skip = x  # 保留输入作为skip

        x = self.conv(x)

        # adj = self.GC(x)
        # x = self.gcn(x, adj)

        ###### 消融实验-3: 有空间卷积 vs. 无空间卷积 ######
        ### (有/无)空间卷积
        x = self.gcn(x, adj)  # 动态图从外而来

        x = x*self.emb + skip

        return x



###### 交互模块 ######
class IDGCN(nn.Module) :


    def __init__(self, channels, diffusion_k, dropout, emb=None) :

        super(IDGCN, self).__init__()

        Conv1 = []
        Conv2 = []
        Conv3 = []
        Conv4 = []
        pad_l = 3  # 数值有何讲究
        pad_r = 3

        k1 = 5     # 数值有何讲究
        k2 = 3     # 数值有何讲究
        Conv1 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)), 
            nn.Conv2d(channels, channels, kernel_size=(1, k1)), 
            nn.LeakyReLU(negative_slope=0.01, inplace=True), 
            nn.Dropout(dropout), 
            nn.Conv2d(channels, channels, kernel_size=(1, k2)), 
            nn.Tanh()
        ]
        Conv2 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)), 
            nn.Conv2d(channels, channels, kernel_size=(1, k1)), 
            nn.LeakyReLU(negative_slope=0.01, inplace=True), 
            nn.Dropout(dropout), 
            nn.Conv2d(channels, channels, kernel_size=(1, k2)), 
            nn.Tanh()
        ]
        Conv3 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)), 
            nn.Conv2d(channels, channels, kernel_size=(1, k1)), 
            nn.LeakyReLU(negative_slope=0.01, inplace=True), 
            nn.Dropout(dropout), 
            nn.Conv2d(channels, channels, kernel_size=(1, k2)), 
            nn.Tanh()
        ]
        Conv4 += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)), 
            nn.Conv2d(channels, channels, kernel_size=(1, k1)), 
            nn.LeakyReLU(negative_slope=0.01, inplace=True), 
            nn.Dropout(dropout), 
            nn.Conv2d(channels, channels, kernel_size=(1, k2)), 
            nn.Tanh()
        ]

        # 多个函数运算对象，串联成的神经网络，其返回的是Module类型的神经网络对象
        self.conv1 = nn.Sequential(*Conv1)
        self.conv2 = nn.Sequential(*Conv2)
        self.conv3 = nn.Sequential(*Conv3)
        self.conv4 = nn.Sequential(*Conv4)

        self.dgcn  = DGCN(channels, diffusion_k, dropout, emb)


    # [B, 2C, N, W] + [B, 2C, N, W] + [B, N, N] + [B, N, N] -- [B, 2C, N, W]
    def forward(self, xl0, xh0, Al, Ah) :       # self.dgcn只有一例，即一套参数

        xl_t    = self.conv1(xl0)               # 时间卷积
        xl_ts   = self.dgcn(xl_t, Al)           # 动态空间卷积
        xl_1    = xl_ts.mul(torch.tanh(xh0))    # 低频，交互高频

        xh_t    = self.conv2(xh0)               # 时间卷积
        xh_ts   = self.dgcn(xh_t, Ah)           # 动态空间卷积
        xh_1    = xh_ts.mul(torch.tanh(xl0))    # 高频，交互低频

        xl_1_t  = self.conv3(xl_1)              # 夹杂了高频成分的，时间卷积
        xl_1_ts = self.dgcn(xl_1_t, Al)         # 夹杂了高频成分的，动态空间卷积
        xl_2     = 0.5*xl_1_ts + 0.5*xh_1        # 低频，吸收高频。注意力吸收？

        xh_1_t  = self.conv4(xh_1)              # 夹杂了低频成分的，时间卷积
        xh_1_ts = self.dgcn(xh_1_t, Ah)         # 夹杂了低频成分的，动态空间卷积
        xh_2     = 0.5*xh_1_ts + 0.5*xl_1        # 高频，吸收低频。注意力吸收？

        return (xl_2, xh_2)



###### 交互树 ######
class IDGCN_Tree(nn.Module) :


    def __init__(self, nodes, windows, channels, diffusion_k, dropout, layer_tree) :

        super().__init__()

        self.layer_tree = layer_tree

        # 作用是仿射？定义的位置。
        self.memory_1   = nn.Parameter(torch.randn(channels, nodes, windows))  # [2C, N, W]
        self.memory_2   = nn.Parameter(torch.randn(channels, nodes, windows))  # [2C, N, W]
        self.memory_3   = nn.Parameter(torch.randn(channels, nodes, windows))  # [2C, N, W]

        self.IDGCN_1 = IDGCN(
            channels       = channels,        # 2C
            diffusion_k    = diffusion_k,     # K
            dropout        = dropout,         # D
            emb            = self.memory_1
        )
        self.IDGCN_2 = IDGCN(
            channels       = channels,        # 2C
            diffusion_k    = diffusion_k,     # K
            dropout        = dropout,         # D
            emb            = self.memory_2
        )
        self.IDGCN_3 = IDGCN(
            channels       = channels,        # 2C
            diffusion_k    = diffusion_k,     # K
            dropout        = dropout,         # D
            emb            = self.memory_3
        )


    def forward(self, xl, xh, Al, Ah) :

        if   self.layer_tree == 1 :
            xl, xh = self.IDGCN_1(xl, xh, Al, Ah)
        elif self.layer_tree == 2 :
            xl, xh = self.IDGCN_1(xl, xh, Al, Ah)
            xl, xh = self.IDGCN_2(xl, xh, Al, Ah)
        elif self.layer_tree == 3 :
            xl, xh = self.IDGCN_1(xl, xh, Al, Ah)
            xl, xh = self.IDGCN_2(xl, xh, Al, Ah)
            xl, xh = self.IDGCN_3(xl, xh, Al, Ah)

        return xl, xh



###### 时间位置编码向量 ######
class TemporalEmbedding(nn.Module) :


    def __init__(self, channels, granularity) :

        super(TemporalEmbedding, self).__init__()

        self.granularity = granularity  # 每天的timestamp的数量
        self.vec_day  = nn.Parameter(torch.empty(granularity, channels))  # [G, C]
        nn.init.xavier_uniform_(self.vec_day)

        self.vec_week = nn.Parameter(torch.empty(7, channels))            # [7, C]
        nn.init.xavier_uniform_(self.vec_week)


    # 利用另外两个维度的信息，生成时间位置编码向量，附着到原输入数据上
    def forward(self, x) :  # [B, W, N, 3]

        emb_day  = x[..., 1]  # -- [B, W, N]
        # print(emb_day[:, :, :]*self.granularity)  # 原本为0～1，*G放大了值，变成整数，成为索引
        # print(self.vec_day.shape)  # [G, C]
        vec_day  = self.vec_day[(emb_day[:, :, :]*self.granularity).type(torch.LongTensor)]
        # 用3D-[B, W, N]在2D-[G, C]的第1D里索引，得到4D-[B, W, N, C]
        # print(vec_day.shape)  # [B, W, N, C]
        vec_day  = vec_day.transpose(1, 2).contiguous()  # -- [B, N, W, C]

        emb_week = x[..., 2]
        vec_week = self.vec_week[(emb_week[:, :, :]).type(torch.LongTensor)]
        vec_week = vec_week.transpose(1, 2).contiguous()

        vec_time = vec_day + vec_week
        vec_time = vec_time.permute(0, 3, 1, 2)  # -- [B, C, N, W]

        return vec_time



###### 总体模型 ######
class STGNN_NN(nn.Module) :


    def __init__(self, device, nodes, windows, horizons, 
                 revin_en, wavelets, level, channels, granularity, 
                 graph_dims, diffusion_k, dropout, layer_tree) :


        super().__init__()


        self.device   = device
        self.revin_en = revin_en  # PEMS的效果不好，series的效果较好
        self.wavelets = wavelets
        self.level    = level


        ######################## 时间位置编码 ########################
        self.T_emb = TemporalEmbedding(channels, granularity)


        ######################## RevIN ########################
        if self.revin_en == True :
            self.revin = RevIN(num_features=nodes, affine=False)  # False ???


        ######################## 语义卷积 ########################
        self.start_conv_l = nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=(1, 1))
        self.start_conv_h = nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=(1, 1))


        ######################## 动态构图 ########################
        self.graph_constructor = graph_constructor(device, nodes, windows, graph_dims, dropout)


        ######################## 动态交互 ########################
        self.tree = IDGCN_Tree(
            nodes          = nodes,           # N
            windows        = windows,         # W
            channels       = channels*2,      # 2C
            diffusion_k    = diffusion_k,     # K
            dropout        = dropout,         # D
            layer_tree     = layer_tree       # L
        )


        ######################## 自动融合 ########################
        self.lamda = nn.Parameter(torch.randn(1))
        self.adp_fusion = Adaptive_Fusion(8, (channels*2)//8)  # h与d，参数可调？


        ######################## 预测结果 ########################
        self.glu        = GLU(channels*2, dropout)
        self.regression = nn.Conv2d(channels*2, horizons, kernel_size=(1, windows))


    def forward(self, x) :  # tensor: [B, 3, N, W]


        ######################## 时间位置编码 ########################
        time_emb = self.T_emb(x.transpose(1, 3))  # [B, 3, N, W] -- [B, W, N, 3] -- [B, C, N, W]


        ######################## 数据规整 ########################
        x_tensor = x[:, 0, :, :].unsqueeze(1)     # -- [B, 1, N, W]
        ######################## RevIN ########################
        if self.revin_en == True :
            x_tensor = self.revin(x_tensor.transpose(1, 3), 'norm').transpose(1, 3)  # -- [B, 1, N, W]
        x_np = x_tensor.detach().cpu().numpy()    # -- [B, 1, N, W]


        ###### 消融实验-1: 小波分解 vs. 奇偶分解 ######
        ### (1)-小波分解
        # xl, xh = disentangle(x_np, self.wavelet, 1)    # 可换小波
        _, xl, xh   = multi_wavelet_disentangle(x_np, self.wavelets, self.level)    # 多小波分解
        ### (2)-奇偶分解
        # xl = np.repeat(x_np[:, :, :, 0::2], 2, -1)  # 从0开始
        # xh = np.repeat(x_np[:, :, :, 1::2], 2, -1)  # 从1开始


        ######################## 语义卷积+嵌入 ########################
        xl = torch.Tensor(xl).to(self.device)  # [B, 1, N, W]
        xh = torch.Tensor(xh).to(self.device)  # [B, 1, N, W]
        xl = torch.cat([self.start_conv_l(xl)] + [time_emb], dim=1)  # [B, 2C, N, W]
        xh = torch.cat([self.start_conv_h(xh)] + [time_emb], dim=1)  # [B, 2C, N, W]


        ######################## 动态构图 ########################
        A = self.graph_constructor(x_tensor)  # [B, 1, N, W] -- [B, N, N]


        ######################## 动态交互 ########################
        xl, xh = self.tree(xl, xh, A, A)


        ###### 消融实验-4: 学习融合 vs. 手动融合 vs. 自动融合 ######
        ### (1)-学习融合
        x_fse  = torch.sigmoid(self.lamda)*xl + (1-torch.sigmoid(self.lamda))*xh
        ### (2)-手动融合
        # lamda  = 0.5  # 超参数可调
        # x_fse  = lamda*xl + (1-lamda)*xh
        ### (3)-自动融合
        # x_fse  = self.adp_fusion(xl, xh)  # 2C维度进入


        ######################## 预测结果 ########################
        gcn        = self.glu(x_fse) + x_fse
        prediction = self.regression(F.relu(gcn)).transpose(1, 3)
        # [B, 2C, N, W] -- [B, H, N, 1] -- [B, 1, N, H]


        ######################## iRevIN ########################
        if self.revin_en == True :
            prediction = self.revin(prediction.transpose(1, 3), 
                                    'denorm').transpose(1, 3)  # -- [B, 1, N, H]


        return prediction, A

