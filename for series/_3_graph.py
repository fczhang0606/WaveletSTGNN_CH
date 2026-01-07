########################
####### 20250402 #######
########################



# 
import torch
from torch import nn

from _3_layer import *
from _3_waveletgraph import *
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class graphFusion(nn.Module) :


    def __init__(self, device, nodes, graph_dims, dropout) :

        super(graphFusion, self).__init__()

        self.device   = device
        self.nodes    = nodes

        # 
        self.lamda    = nn.Parameter(torch.randn(1))

        # 
        self.norm_adj_sta   = nn.LayerNorm(nodes)       # 静态图
        self.norm_adj_dyn   = nn.LayerNorm(nodes)       # 变化图

        self.norm_vec_fse   = nn.LayerNorm(graph_dims)  # 融合特征
        self.vec_fsed_prs   = nn.Parameter(torch.zeros(size=(nodes, graph_dims)))  # [N, G]
        nn.init.xavier_uniform_(self.vec_fsed_prs.data, gain=1.414)
        self.norm_adj_fse   = nn.LayerNorm(nodes)       # 融合图

        # 多头注意力。参数做成输入可调?
        self.head_num = 4
        self.head_dim = 8
        self.D        = self.head_num * self.head_dim   # 32
        self.dropout  = dropout
        self.temperature  = 0.5

        # G ~ 32
        self.query = fc_layer(in_channels=graph_dims, out_channels=self.D, need_layer_norm=False)
        self.key   = fc_layer(in_channels=graph_dims, out_channels=self.D, need_layer_norm=False)

        self.norm_adj_att = nn.LayerNorm(nodes)         # 注意力图
        # self.norm_adj_wgt = SELayer(channel=3)          # 带权重图

        # # 
        # self.wgt_mlp1_end = nn.Conv2d(in_channels=3, out_channels=self.D, kernel_size=(1, 1))  # 3 ~ D
        # self.wgt_mlp2_end = nn.Conv2d(in_channels=self.D, out_channels=2, kernel_size=(1, 1))  # D ~ 2


    def _calculate_random_walk_matrix(self, adj_mx) :  # [B, N, N]

        adj_eye = torch.eye(int(adj_mx.shape[1])).to(self.device)\
            .unsqueeze(0).repeat(adj_mx.shape[0], 1, 1)
        adj_mx  = adj_mx + adj_eye  # [B, N, N]

        d     = torch.sum(adj_mx, -1)  # [B, N]
        d_inv = 1. / d
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(self.device), d_inv)  # [B, N]

        d_mat = []
        for i in range(adj_mx.shape[0]) :  # B
            d_mat.append(torch.diag(d_inv[i]))

        d_mat_adj      = torch.cat(d_mat, dim=0).reshape(-1, self.nodes, self.nodes)  # [B, N, N]
        random_walk_mx = torch.einsum('bnn, bnm -> bnm', d_mat_adj, adj_mx)

        return random_walk_mx


    # 构建矩阵
    def forward(self, vec_emb, basic_similarities) :
        # [N, G] + [B, N, N]

        batch_size = basic_similarities.shape[0]  # B
        nodes      = basic_similarities.shape[1]  # N

        # 静态图
        adj_sta = self.norm_adj_sta(torch.mm(vec_emb, vec_emb.transpose(1, 0)))  # [N, N]
        adj_sta = adj_sta.unsqueeze(0).repeat(batch_size, 1, 1)                  # [B, N, N]

        '''
        # 变化图
        adj_dyn = self.norm_adj_dyn(torch.einsum('bnd, bdm -> bnm', 
                                                 vec_ftr, vec_ftr.transpose(1, 2)))  # [B, N, N]

        # 融合特征
        vec_fsed = self.norm_vec_fse(vec_fse)
        # [B, N, G] + [N, G] -- [B, N, G]
        vec_fsed = torch.einsum('bnd, nl -> bnl', vec_fsed, self.vec_fsed_prs) \
                   + vec_fsed  # [B, N, G]  # + vec_垫底
        # 融合图
        adj_fse  = torch.einsum('bnd, bdm -> bnm', vec_fsed, vec_fsed.transpose(1, 2))  # [B, N, N]
        adj_fse  = self.norm_adj_fse(adj_fse)


        # 融合特征的自注意力图
        vec_fsed = vec_fsed.unsqueeze(1).transpose(1, -1)  # -- [B, G, N, 1]
        query    = self.query(vec_fsed)                    # -- [B, 32, N, 1]
        key      = self.key  (vec_fsed)                    # -- [B, 32, N, 1]

        # Q、K的重新展开
        query = query.squeeze(-1).contiguous()\
            .view(batch_size, self.head_num, self.head_dim, nodes).transpose(-1, -2)  # [B, 4, N, 8]
        key   = key  .squeeze(-1).contiguous()\
            .view(batch_size, self.head_num, self.head_dim, nodes)                    # [B, 4, 8, N]

        attention = torch.einsum('bhnd, bhdu -> bhnu', query, key)                    # [B, 4, N, N]
        attention /= (self.head_dim ** 0.5)  # 
        attention = F.dropout(attention, self.dropout, training=self.training)        # self.training ?
        adj_att   = self.norm_adj_att(torch.sum(attention, dim=1)) + adj_fse          # [B, N, N]
        # 没有V？+adj_fse？
        '''


        # 融合矩阵，可以改进？？？
        adj = torch.sigmoid(self.lamda)*adj_sta + \
            (1-torch.sigmoid(self.lamda))*basic_similarities  # [B, N, N]
        # adj = basic_similarities  # [B, N, N]
        adj = F.softmax(adj, dim=1)         # 值域[0, 1]，行和为1


        # 另外的融合方式
        # adj_wgt = torch.stack(  # [B, 3, N, N]  # torch.cat
        #     [adj_sta.unsqueeze(1), adj_dyn.unsqueeze(1), adj_att.unsqueeze(1)], dim=1).squeeze()
        # adj_wgt = self.norm_adj_wgt(adj_wgt) + adj_wgt  # SE  # [B, 3, N, N]
        # adj_end = F.dropout(self.wgt_mlp2_end(torch.relu(self.wgt_mlp1_end(adj_wgt))), 
        #                     self.dropout, training=self.training)  # [B, 2, N, N]
        # adj_end = adj_end.permute(0, 2, 3, 1).reshape(batch_size, nodes*nodes, 2)  # [B, N2, 2]

        # # Get discrete graph structures
        # adj_tmp, _ = gumbel_softmax(adj_end, temperature=self.temperature, hard=True)  # [B, N2, 2]
        # adj  = adj_tmp[..., 0].clone().reshape(batch_size, nodes, nodes)  # [B, N, N]
        # mask = torch.eye(nodes, nodes).bool().to(self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        # adj.masked_fill_(mask, 0)

        # # Normalized graph matrices
        # adj = self._calculate_random_walk_matrix(adj)  # [B, N, N]


        ###### 消融实验-2: 动态图 vs. 静态图 ######
        ### (1)-动态图
        return adj
        ### (2)-静态图
        # return adj_sta



### 构图函数
class graph_constructor(nn.Module) :


    def __init__(self, device, nodes, windows, 
                 graph_dims, dropout) :

        super(graph_constructor, self).__init__()

        self.device      = device
        self.nodes       = nodes

        self.E           = nn.Embedding(nodes, graph_dims)  # [N, G]

        # self.trans_ftr   = nn.Conv2d(1, graph_dims, kernel_size=(1, windows))  # 1~G
        # self.norm_ftr    = nn.LayerNorm(graph_dims)

        # self.gate_Fusion = gatedFusion(device, graph_dims)  # layer.py

        self.wavelet_adjs= WaveletGraph(device=device, wave='db4', J=4)  # [B, N, J+1] -- [B, N, N]

        self.graph_Fusion= graphFusion(device, nodes, graph_dims, dropout)


    ### 节点特征作为输入，参与构图。过往时间步的信息也有参考融合。
    def forward(self, x) :  # [B, 1, N, W]

        idx     = torch.arange(self.nodes).to(self.device)  # [N]
        vec_emb = self.E(idx)  # [N, G]

        # [B, 1, N, W] -- [B, G, N, 1] -- [B, G, N] -- [B, N, G]
        # vec_ftr = self.norm_ftr(self.trans_ftr(x).squeeze(-1).transpose(1, 2))

        # -- [B, N, G]
        # vec_fse = self.gate_Fusion(x.shape[0], vec_emb, vec_ftr) + vec_emb.unsqueeze(0).repeat(x.shape[0], 1, 1)

        # -- [B, N, N]
        _, wavelet_adjs, _ = self.wavelet_adjs.wavelet_adjs(x)

        # 
        adj     = self.graph_Fusion(vec_emb, wavelet_adjs['cosine'])  # 选择

        return adj

