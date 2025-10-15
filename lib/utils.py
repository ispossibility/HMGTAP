import numpy as np
import pandas as pd
import torch
import sys
import os
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, jaccard
from numpy.linalg import norm
from scipy.stats import pearsonr
from tqdm import tqdm

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from lib.metrics import mask_evaluation_np


def compute_pearson_similarity_optimized(graph_x):
    batch_size, seq_len, feature_dim, n = graph_x.shape
    graph_x = graph_x[:, :seq_len - 4, :, :]
    similarity_matrix = np.zeros((batch_size, n, n))
    for b in range(batch_size):
        for i in tqdm(range(n)):
            for j in range(i + 1, n):
                seq_i = graph_x[b, :, :, i]
                seq_j = graph_x[b, :, :, j]
                max_pearson = -1

                for d in range(feature_dim):
                    seq_i_d = seq_i[:, d]
                    seq_j_d = seq_j[:, d]
                    if np.all(seq_i_d == 0) and np.any(seq_j_d != 0):
                        pearson_sim = 0
                    elif np.all(seq_j_d == 0) and np.any(seq_i_d != 0):
                        pearson_sim = 0
                    else:
                        pearson_sim, _ = pearsonr(seq_i_d, seq_j_d)
                        pearson_sim = pearson_sim if not np.isnan(pearson_sim) else 0
                    max_pearson = max(max_pearson, pearson_sim)
                similarity_matrix[b, i, j] = max_pearson
                similarity_matrix[b, j, i] = max_pearson

    return similarity_matrix


def compute_dtw_similarity(graph_x):
    batch_size, seq_len, feature_dim, n = graph_x.shape
    similarity_matrix = np.zeros((batch_size, n, n))
    for b in range(batch_size):
        for i in tqdm(range(n)):
            for j in range(n):
                if i == j:
                    similarity_matrix[b, i, j] = 1.0
                else:
                    seq_i = graph_x[b, :seq_len - 4, :, i]
                    seq_j = graph_x[b, :seq_len - 4, :, j]
                    min_dtw_distance = float('inf')
                    for d in range(feature_dim):
                        seq_i_d = np.asarray(seq_i[:, d]).flatten()
                        seq_j_d = np.asarray(seq_j[:, d]).flatten()
                        if np.all(seq_i_d == 0) and np.all(seq_j_d == 0):
                            dtw_distance = float('inf')
                        else:
                            seq_i_d = seq_i_d.flatten()
                            seq_j_d = seq_j_d.flatten()
                            dtw_distance, _ = fastdtw(seq_i_d, seq_j_d, dist=lambda u, v: np.linalg.norm(u - v))
                        min_dtw_distance = min(min_dtw_distance, dtw_distance)
                    similarity_matrix[b, i, j] = 1 / (1 + min_dtw_distance) if min_dtw_distance < float('inf') else 0
    return similarity_matrix


def construct_hypergraph(max_sequence, n):
    hypergraph_matrix = np.zeros((n, 1))
    for idx, node in enumerate(max_sequence):
        hypergraph_matrix[node, 0] = 1
    return hypergraph_matrix


class Scaler_NYC:
    def __init__(self, train):
        """ NYC Max-Min

        Arguments:
            train {np.ndarray} -- shape(T, D, W, H)
        """
        train_temp = np.transpose(train, (0, 2, 3, 1)).reshape((-1, train.shape[1]))
        self.max = np.max(train_temp, axis=0)
        self.min = np.min(train_temp, axis=0)

    def transform(self, data):
        """norm train，valid，test

        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)

        Returns:
            {np.ndarray} -- shape(T, D, W, H)
        """
        T, D, W, H = data.shape
        data = np.transpose(data, (0, 2, 3, 1)).reshape((-1, D))
        data[:, 0] = (data[:, 0] - self.min[0]) / (self.max[0] - self.min[0])
        data[:, 33:40] = (data[:, 33:40] - self.min[33:40]) / (self.max[33:40] - self.min[33:40])
        data[:, 40] = (data[:, 40] - self.min[40]) / (self.max[40] - self.min[40])
        data[:, 46] = (data[:, 46] - self.min[46]) / (self.max[46] - self.min[46])
        data[:, 47] = (data[:, 47] - self.min[47]) / (self.max[47] - self.min[47])
        return np.transpose(data.reshape((T, W, H, -1)), (0, 3, 1, 2))

    def inverse_transform(self, data):
        """
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)

        Returns:
            {np.ndarray} --  shape (T, D, W, H)
        """
        return data * (self.max[0] - self.min[0]) + self.min[0]


class Scaler_Chi:
    def __init__(self, train):
        """Chicago Max-Min

        Arguments:
            train {np.ndarray} -- shape(T, D, W, H)         D 是特征维度
        """
        # 将输入数组的维度从(T, D, W, H)转换为(T, W, H, D),然后通过 reshape 方法将其重塑为一个形状二维数组，即将时间和空间维度展开为一维。
        train_temp = np.transpose(train, (0, 2, 3, 1)).reshape((-1, train.shape[1]))
        self.max = np.max(train_temp, axis=0)
        self.min = np.min(train_temp, axis=0)

    def transform(self, data):
        """norm train，valid，test

        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)

        Returns:
            {np.ndarray} -- shape(T, D, W, H)
        """
        # 然后，对特定的特征（索引为 0、33、39 和 40 的特征）进行最大最小归一化
        T, D, W, H = data.shape
        data = np.transpose(data, (0, 2, 3, 1)).reshape((-1, D))  # (T*W*H,D)
        data[:, 0] = (data[:, 0] - self.min[0]) / (self.max[0] - self.min[0])
        data[:, 33] = (data[:, 33] - self.min[33]) / (self.max[33] - self.min[33])
        data[:, 39] = (data[:, 39] - self.min[39]) / (self.max[39] - self.min[39])
        data[:, 40] = (data[:, 40] - self.min[40]) / (self.max[40] - self.min[40])
        # 返回的形状重塑为(T, D, W, H)
        return np.transpose(data.reshape((T, W, H, -1)), (0, 3, 1, 2))

    def inverse_transform(self, data):
        """
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)

        Returns:
            {np.ndarray} --  shape(T, D, W, H)
        """
        return data * (self.max[0] - self.min[0]) + self.min[0]


def mask_loss(predicts, labels, region_mask, data_type="nyc"):
    """

    Arguments:
        predicts {Tensor} -- predict，(batch_size,pre_len,W,H)
        labels {Tensor} -- label，(batch_size,pre_len,W,H)
        region_mask {np.array} -- mask matrix，(W,H)
        data_type {str} -- nyc/chicago

    Returns:
        {Tensor} -- MSELoss,(1,)
    """
    batch_size, pre_len, _, _ = predicts.shape
    region_mask = torch.from_numpy(region_mask).to(predicts.device)
    region_mask /= region_mask.mean()
    loss = ((labels - predicts) * region_mask) ** 2
    if data_type == 'nyc':
        ratio_mask = torch.zeros(labels.shape).to(predicts.device)
        index_1 = labels <= 0
        index_2 = (labels > 0) & (labels <= 0.04)
        index_3 = (labels > 0.04) & (labels <= 0.08)
        index_4 = labels > 0.08
        ratio_mask[index_1] = 0.05
        ratio_mask[index_2] = 0.2
        ratio_mask[index_3] = 0.25
        ratio_mask[index_4] = 0.5
        loss *= ratio_mask
    elif data_type == 'chicago':
        ratio_mask = torch.zeros(labels.shape).to(predicts.device)
        index_1 = labels <= 0
        index_2 = (labels > 0) & (labels <= 1 / 17)
        index_3 = (labels > 1 / 17) & (labels <= 2 / 17)
        index_4 = labels > 2 / 17
        ratio_mask[index_1] = 0.05
        ratio_mask[index_2] = 0.2
        ratio_mask[index_3] = 0.25
        ratio_mask[index_4] = 0.5
        loss *= ratio_mask
    return torch.mean(loss)


@torch.no_grad()
def compute_loss(net, dataloader, risk_mask, final_hypergraph_matrix,
                 grid_node_map, global_step, device,
                 data_type='nyc'):
    """compute val/test loss

    Arguments:
        net {Molde} -- model
        dataloader {DataLoader} -- val/test dataloader
        risk_mask {np.array} -- mask matrix，shape(W,H)
        road_adj  {np.array} -- road adjacent matrix，shape(N,N)
        risk_adj  {np.array} -- risk adjacent matrix，shape(N,N)
        poi_adj  {np.array} -- poi adjacent matrix，shape(N,N)
        global_step {int} -- global_step
        device {Device} -- GPU

    Returns:
        np.float32 -- mean loss
    """
    net.eval()
    temp = []
    for feature, target_time, graph_feature, hypergraphs_array, label in dataloader:
        feature, target_time, graph_feature, hypergraphs_array, label = feature.to(device), target_time.to(
            device), graph_feature.to(device), hypergraphs_array.to(device), label.to(device)
        l = mask_loss(
            net(feature, target_time, graph_feature, final_hypergraph_matrix, hypergraphs_array, grid_node_map), label,
        temp.append(l.cpu().item())
    loss_mean = sum(temp) / len(temp)
    return loss_mean


@torch.no_grad()
def predict_and_evaluate(net, dataloader, risk_mask, final_hypergraph_matrix,
                         grid_node_map, global_step, scaler, device):
    """predict val/test, return metrics

    Arguments:
        net {Model} -- model
        dataloader {DataLoader} -- val/test dataloader
        risk_mask {np.array} -- mask matrix，shape(W,H)
        road_adj  {np.array} -- road adjacent matrix，shape(N,N)
        risk_adj  {np.array} -- risk adjacent matrix，shape(N,N)
        poi_adj  {np.array} -- poi adjacent matrix，shape(N,N)
        global_step {int} -- global_step
        scaler {Scaler} -- record max and min
        device {Device} -- GPU

    Returns:
        np.float32 -- RMSE，Recall，MAP
        np.array -- label and pre，shape(num_sample,pre_len,W,H)

    """
    net.eval()
    prediction_list = []
    label_list = []
    for feature, target_time, graph_feature, hypergraphs_array, label in dataloader:
        feature, target_time, graph_feature, hypergraphs_array, label = feature.to(device), target_time.to(
            device), graph_feature.to(device), hypergraphs_array.to(device), label.to(device)
        prediction_list.append(net(feature, target_time, graph_feature, final_hypergraph_matrix, hypergraphs_array,
                                   grid_node_map).cpu().numpy())
        label_list.append(label.cpu().numpy())
    prediction = np.concatenate(prediction_list, 0)
    label = np.concatenate(label_list, 0)
    inverse_trans_pre = scaler.inverse_transform(prediction)
    inverse_trans_label = scaler.inverse_transform(label)
    rmse_, recall_, map_, mae, mape, r_2 = mask_evaluation_np(inverse_trans_label, inverse_trans_pre, risk_mask, 0)
    return rmse_, recall_, map_, mae, mape, r_2, inverse_trans_pre, inverse_trans_label
