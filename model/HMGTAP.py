import math

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import json

import sys
import os

from torch.nn import Parameter

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


class TransformerModule(nn.Module):
    def __init__(self, feature_size, num_heads, ff_dim, dropout=0.1):
        super(TransformerModule, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_size, num_heads=num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(feature_size, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, feature_size)
        )
        self.layernorm1 = nn.LayerNorm(feature_size)
        self.layernorm2 = nn.LayerNorm(feature_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G, G1):
        bs, _, _ = x.shape
        time_stamp, _, _ = G1.shape
        G = torch.from_numpy(G).to(dtype=torch.float32, device=x.device)
        G = G.unsqueeze(0).expand(time_stamp, -1, -1)
        # G1 = torch.from_numpy(G1).to(dtype=torch.float32, device=x.device)
        G_combined = torch.cat([G, G1], dim=-1)
        L_list = []
        for i in range(time_stamp):
            G_i = G_combined[i]
            Dv = torch.diag(G_i.sum(dim=1) + 1e-6)
            De = torch.diag(G_i.sum(dim=0) + 1e-6)
            Dv_inv_sqrt = torch.linalg.inv(torch.sqrt(Dv))
            Dv_inv_sqrt[torch.isinf(Dv_inv_sqrt)] = 0
            L = Dv_inv_sqrt @ G_i @ torch.linalg.inv(De) @ G_i.T @ Dv_inv_sqrt
            L_list.append(L)

        L = torch.stack(L_list, dim=0)
        L = L.unsqueeze(1).expand(-1, 28, -1, -1)
        L = L.reshape(bs, L.size(-2), L.size(-1))
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = L.matmul(x)
        return x

class RegionModule(nn.Module):
    def __init__(self,grid_in_channel,num_of_gru_layers,seq_len,
                gru_hidden_size,num_of_target_time_feature):

        super(RegionModule,self).__init__()
        self.grid_conv = nn.Sequential(
            nn.Conv2d(in_channels=grid_in_channel,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=1,kernel_size=3,padding=1),
            nn.ReLU(),
        )

        self.grid_gru = nn.GRU(grid_in_channel,gru_hidden_size,num_of_gru_layers,batch_first=True)
        self.grid_att_fc1 = nn.Linear(in_features=gru_hidden_size,out_features=1)
        self.grid_att_fc2 = nn.Linear(in_features=num_of_target_time_feature,out_features=seq_len)
        self.grid_att_bias = nn.Parameter(torch.zeros(1))
        self.grid_att_softmax = nn.Softmax(dim=-1)


    def forward(self,grid_input,target_time_feature):
        batch_size,T,D,W,H = grid_input.shape
        grid_input = grid_input.view(-1,D,W,H)
        conv_output = self.grid_conv(grid_input)
        grid_output = conv_output.view(batch_size, T, -1, W, H)
        return grid_output

class HyperModule(nn.Module):
    def __init__(self,num_of_graph_feature,nums_of_graph_filters,
                seq_len,num_of_gru_layers,gru_hidden_size,
                num_of_target_time_feature,north_south_map,west_east_map):
        super(HyperModule,self).__init__()
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map
        self.dropout = 0.1
        self.hgc1 = HGNN_conv(num_of_graph_feature, nums_of_graph_filters[0])
        self.hgc2 = HGNN_conv(nums_of_graph_filters[0], nums_of_graph_filters[1])
        self.hgc3 = HGNN_conv(nums_of_graph_filters[1], 1)
        self.graph_gru = nn.GRU(num_of_graph_feature, gru_hidden_size, 1, batch_first=True)
        self.graph_att_fc1 = nn.Linear(in_features=gru_hidden_size,out_features=1)
        self.graph_att_fc2 = nn.Linear(in_features=num_of_target_time_feature,out_features=seq_len)
        self.graph_att_bias = nn.Parameter(torch.zeros(1))
        self.graph_att_softmax = nn.Softmax(dim=-1)


    def forward(self,graph_feature,final_hypergraph_matrix,hypergraphs_array,
                target_time_feature,grid_node_map):
        batch_size,T,D1,N = graph_feature.shape
        road_graph_output = graph_feature.view(-1,D1,N).permute(0,2,1).contiguous()
        x = F.relu(self.hgc1(road_graph_output, final_hypergraph_matrix, hypergraphs_array))
        x = F.dropout(x, self.dropout)
        for _ in range(3):
            x = self.hgc2(x, final_hypergraph_matrix, hypergraphs_array)
            x = F.dropout(x, self.dropout)
        x = self.hgc3(x, final_hypergraph_matrix, hypergraphs_array)
        graph_output = x
        grid_node_map_tmp = torch.from_numpy(grid_node_map)\
                            .to(graph_feature.device)\
                            .repeat(batch_size*T,1,1)
        graph_output = torch.bmm(grid_node_map_tmp,graph_output)\
                            .permute(0,2,1)\
                            .view(batch_size,T,-1,self.north_south_map,self.west_east_map)
        return graph_output


class HMGTAP(nn.Module):
    def __init__(self,grid_in_channel,num_of_gru_layers,seq_len,pre_len,
                gru_hidden_size,num_of_target_time_feature,
                num_of_graph_feature,nums_of_graph_filters,
                north_south_map,west_east_map):
        super(HMGTAP,self).__init__()
        self.north_south_map = north_south_map
        self.west_east_map = west_east_map
        self.rm = RegionModule(grid_in_channel,num_of_gru_layers,seq_len,
                                        gru_hidden_size,num_of_target_time_feature)
        self.hm = HyperModule(num_of_graph_feature,nums_of_graph_filters,
                                        seq_len,num_of_gru_layers,gru_hidden_size,
                                        num_of_target_time_feature,north_south_map,west_east_map)
        
        fusion_channel = 1  # 1 2 3
        self.grid_weigth = nn.Conv2d(in_channels=gru_hidden_size,out_channels=fusion_channel,kernel_size=1)
        self.graph_weigth = nn.Conv2d(in_channels=gru_hidden_size,out_channels=fusion_channel,kernel_size=1)
        # self.grid_input_weigth = nn.Conv2d(in_channels=48, out_channels=fusion_channel, kernel_size=1)
        self.grid_input_weigth = nn.Conv2d(in_channels=41, out_channels=fusion_channel, kernel_size=1)
        self.transformer_module = TransformerModule(1200,8,2048)
        self.GRU_module = nn.GRU(input_size=1200, hidden_size=400)
        self.output_layer = nn.Linear(seq_len*north_south_map*west_east_map,pre_len*north_south_map*west_east_map)


    def forward(self,grid_input,target_time_feature,graph_feature,
                final_hypergraph_matrix,hypergraphs_array,grid_node_map):
        batch_size,T,D,_,_ =grid_input.shape
        _, _, D1, _ = graph_feature.shape
        grid_output = self.rm(grid_input,target_time_feature)
        graph_output = self.hm(graph_feature,final_hypergraph_matrix,hypergraphs_array,
                                        target_time_feature,grid_node_map)
        grid_output = grid_output.reshape(batch_size, T, 1, -1)
        graph_output = graph_output.reshape(batch_size, T, 1, -1)
        fusion_out = torch.cat([grid_output, graph_output], dim=2)
        # grid_input = grid_input.reshape(batch_size, T, 48, -1).permute(0, 2, 1, 3)
        grid_input = grid_input.reshape(batch_size,T,41,-1).permute(0,2,1,3)
        grid_input = self.grid_input_weigth(grid_input)
        grid_input = grid_input.permute(0, 2, 1, 3)
        fusion_out = torch.cat([fusion_out, grid_input], dim=2)
        fusion_output = self.transformer_module(fusion_out.reshape(batch_size, T, -1).permute(1,0,2))
        fusion_output, hn = self.GRU_module(fusion_output)
        fusion_output = F.relu(fusion_output)
        fusion_output = fusion_output.permute(1,0,2)
        fusion_output = fusion_output.reshape(batch_size,-1)
        final_output = self.output_layer(fusion_output)\
                            .view(batch_size,-1,self.north_south_map,self.west_east_map)
        return final_output
