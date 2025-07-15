import numpy as np
import pickle as pkl
import configparser
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from lib.utils import Scaler_NYC,Scaler_Chi

def generate_sin_cos_encoding(seq_len, feature_dim):

    position = np.arange(seq_len, dtype=np.float32)[:, np.newaxis]
    div_term = np.exp(np.arange(0, feature_dim, 2) * (-np.log(10000.0) / feature_dim))
    pos_encoding = np.zeros((seq_len, feature_dim), dtype=np.float32)
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)

    return pos_encoding

def apply_position_encoding(data, day_len=24, week_len=7, feature_dim=2):

    batch_size, feature_dim_in, w, h = data.shape
    day_indices = np.arange(batch_size) % day_len
    week_indices = np.arange(batch_size) // day_len % week_len
    daily_encoding = generate_sin_cos_encoding(day_len, feature_dim)
    daily_encoding = daily_encoding[day_indices]
    weekly_encoding = generate_sin_cos_encoding(week_len, feature_dim)
    weekly_encoding = weekly_encoding[week_indices]
    time_encoding = daily_encoding + weekly_encoding
    time_encoding = time_encoding[:, :, np.newaxis, np.newaxis]
    time_encoding = np.tile(time_encoding, (1, 1, w, h))
    data_with_pe = data.copy()
    # data_with_pe[:, [46, 47], :, :] += time_encoding
    data_with_pe[:, [39, 40], :, :] += time_encoding
    return data_with_pe

def split_and_norm_data_time(all_data,
                        train_rate = 0.6,
                        valid_rate = 0.2,
                        recent_prior=3,
                        week_prior=4,
                        one_day_period=24,
                        days_of_week=7,
                        pre_len=1):
    num_of_time,channel,_,_ = all_data.shape
    train_line, valid_line = int(num_of_time * train_rate), int(num_of_time * (train_rate+valid_rate))
    for index,(start,end) in enumerate(((0,train_line),(train_line,valid_line),(valid_line,num_of_time))):
        if index == 0:
            if channel == 48:
                scaler = Scaler_NYC(all_data[start:end,:,:,:])
            if channel == 41:
                scaler = Scaler_Chi(all_data[start:end,:,:,:])
        norm_data = scaler.transform(all_data[start:end,:,:,:])
        norm_data = apply_position_encoding(norm_data, day_len=one_day_period, week_len=days_of_week,
                                            feature_dim=2)
        X,Y,target_time = [],[],[]
        high_X,high_Y,high_target_time = [],[],[]
        for i in range(len(norm_data)-week_prior*days_of_week*one_day_period-pre_len+1):
            t = i+week_prior*days_of_week*one_day_period
            label = norm_data[t:t+pre_len,0,:,:]
            period_list = []
            for week in range(week_prior):
                period_list.append(i+week*days_of_week*one_day_period)
            for recent in list(range(1,recent_prior+1))[::-1]:
                period_list.append(t-recent)
            feature = norm_data[period_list,:,:,:]
            X.append(feature)
            Y.append(label)
            target_time.append(norm_data[t,1:33,0,0])
            if list(norm_data[t,1:25,0,0]).index(1) in high_fre_hour:
                high_X.append(feature)
                high_Y.append(label)
                high_target_time.append(norm_data[t,1:33,0,0])
        yield np.array(X),np.array(Y),np.array(target_time),np.array(high_X),np.array(high_Y),np.array(high_target_time),scaler


def normal_and_generate_dataset_time(
        all_data_filename,
        train_rate=0.6,
        valid_rate=0.2,
        recent_prior=3,
        week_prior=4,
        one_day_period=24,
        days_of_week=7,
        pre_len=1):
    all_data = pkl.load(open(all_data_filename,'rb')).astype(np.float32)
    for i in split_and_norm_data_time(all_data,
                        train_rate = train_rate,
                        valid_rate = valid_rate,
                        recent_prior = recent_prior,
                        week_prior = week_prior,
                        one_day_period = one_day_period,
                        days_of_week = days_of_week,
                        pre_len = pre_len):
        yield i 

def get_mask(mask_path):
    """
    Arguments:
        mask_path {str} -- mask filename
    
    Returns:
        {np.array} -- mask matrix，维度(W,H)
    """
    mask = pkl.load(open(mask_path,'rb')).astype(np.float32)
    return mask

def get_adjacent(adjacent_path):
    """
    Arguments:
        adjacent_path {str} -- adjacent matrix path
    
    Returns:
        {np.array} -- shape:(N,N)
    """
    adjacent = pkl.load(open(adjacent_path,'rb')).astype(np.float32)
    return adjacent

def get_grid_node_map_maxtrix(grid_node_path):
    """
    Arguments:
        grid_node_path {str} -- filename
    
    Returns:
        {np.array} -- shape:(W*H,N)
    """
    grid_node_map = pkl.load(open(grid_node_path,'rb')).astype(np.float32)
    return grid_node_map 
