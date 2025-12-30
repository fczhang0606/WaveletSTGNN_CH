########################
####### 20250402 #######
########################



# 
import torch
from torch import nn
import torch.nn.functional as F



###### 前馈网络 ######
class fc_layer(nn.Module) :


    def __init__(self, in_channels, out_channels, need_layer_norm) :

        super(fc_layer, self).__init__()

        self.need_layer_norm = need_layer_norm

        self.param_w = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))  # [G, D]
        nn.init.xavier_uniform_(self.param_w.data, gain=1.414)

        self.layer_norm = nn.LayerNorm(out_channels)


    def forward(self, input) :  # [B, G, N, 1]

        ### 分支完全一样？

        # ([B, G, N, 1]--[B, 1, N, G]) ** [G, D] -- [B, 1, N, D] -- [B, D, N, 1]
        if self.need_layer_norm :
            result = F.leaky_relu(torch.einsum('bani, io -> bano', 
                                               [input.transpose(1, -1), self.param_w])).transpose(1, -1)
        else :
            result = F.leaky_relu(torch.einsum('bani, io -> bano', 
                                               [input.transpose(1, -1), self.param_w])).transpose(1, -1)

        return result



### 变换？？？
class SELayer(nn.Module) :


    def __init__(self, channel, reduction=4) :  # 3 + 4

        super(SELayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel*reduction, bias=False),  # 3 -> 12
            nn.ReLU(inplace=True), 
            nn.Linear(channel*reduction, channel, bias=False),  # 12 -> 3
            nn.Sigmoid()
        )


    def forward(self, x) :  # [B, 3, N, N]

        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # [B, 3]
        y = self.fc(y).view(b, c, 1, 1)  # [B, 3] -- [B, 12] -- [B, 3, 1, 1] or [B, 3, N, N]

        return x*y.expand_as(x)



###### 门控融合 ######
class gatedFusion(nn.Module) :


    def __init__(self, device, graph_dims) :  # G

        super(gatedFusion, self).__init__()

        self.device = device

        self.z_emb = nn.Linear(in_features=graph_dims, out_features=graph_dims)  # G ~ G
        self.z_ftr = nn.Parameter(torch.zeros(size=(graph_dims, graph_dims)))    # [G, G]
        nn.init.xavier_uniform_(self.z_ftr, gain=1.414)

        self.r_emb = nn.Linear(in_features=graph_dims, out_features=graph_dims)
        self.r_ftr = nn.Linear(in_features=graph_dims, out_features=graph_dims)

        self.h_emb = nn.Linear(in_features=graph_dims, out_features=graph_dims)
        self.h_ftr = nn.Linear(in_features=graph_dims, out_features=graph_dims)


    # GRU实现，两点可以商榷
    def forward(self, batch_size, vec_emb, vec_ftr) :
        # B + [N, G] + [B, N, G]

        # h(t-1) = vec_emb
        # x(t)   = vec_ftr

        # +vec_合适么？？？相当于加了bias吧
        z_emb = self.z_emb(vec_emb) + vec_emb                   # [N, G]
        z_emb = z_emb.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, N, G]  # 先变形再复制，保证一致
        z_ftr = torch.einsum('bnd, dd -> bnd', [vec_ftr, self.z_ftr]) + vec_ftr  # [B, N, G]
        z = torch.sigmoid(z_emb + z_ftr)

        r_emb = self.r_emb(vec_emb).unsqueeze(0).repeat(batch_size, 1, 1) # 先变形再复制，保证一致
        r_ftr = self.r_ftr(vec_ftr)
        r = torch.sigmoid(r_emb + r_ftr)

        # h_emb = self.h_emb(r*(vec_emb.unsqueeze(0).repeat(batch_size, 1, 1)))  # 没有一致性了
        h_emb = r*(self.h_emb(vec_emb).unsqueeze(0).repeat(batch_size, 1, 1))  # 半一致性
        h_ftr = self.h_ftr(vec_ftr)
        h = torch.tanh(h_emb + h_ftr)

        res = torch.add(torch.mul(torch.ones(z.size()).to(self.device)-z, h), z*vec_emb)
        # res = torch.add(torch.mul(torch.ones(z.size()).to(self.device)-z, vec_emb), z*h)

        return res  # [B, N, G]



### https://blog.csdn.net/m0_51098495/article/details/140135477
### https://openreview.net/pdf?id=cGDAkQo1C0p
### https://github.com/ts-kim/RevIN
class RevIN(nn.Module) :


    def __init__(self, num_features:int, eps=1e-5, affine=True) :  # affine=False

        super(RevIN, self).__init__()

        self.num_features = num_features
        self.eps          = eps
        self.affine       = affine
        if self.affine :
            self._init_params()


    def forward(self, x, mode:str) :

        if mode == 'norm' :
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm' :
            x = self._denormalize(x)
        else : raise NotImplementedError

        return x


    def _init_params(self) :
        # initialize RevIN params: (C, )
        self.affine_weight = nn.Parameter(torch.ones (self.num_features))  # 数据升维？
        self.affine_bias   = nn.Parameter(torch.zeros(self.num_features))  # 数据升维？


    def _get_statistics(self, x) :
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean  = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()


    def _normalize(self, x) :
        x = x - self.mean
        x = x / self.stdev
        if self.affine :
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x


    def _denormalize(self, x) :
        if self.affine :
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

