import tensorflow as tf
import numpy as np
import json

data_dir = 'data/'
class DM(object):
    def __init__(self):   
        feature = np.load(data_dir+'cnstock_features_minute.npy')
        with open(data_dir+'stock_tag.json', 'r') as f:
            self.stock_tag = json.load(f)  
        self.stock_num, self.date, self.minute = feature[:, :, :, 0].shape
        self.price = feature[:, :, :, 0].reshape(self.stock_num, -1).T
        self.volume = feature[:, :, :, 1].reshape(self.stock_num, -1).T

    def input_range(self, start_date, end_date, time_span, stocks='all'):
        if stocks == 'all':
            self.stock_idx = np.arange(self.stock_num)
        else:
            self.stock_idx = np.array([self.stock_tag['stock_map'].index(stock) for stock in stocks])        
        self.s_idx = self.stock_tag['date_stamp'].index(start_date)
        self.e_idx = self.stock_tag['date_stamp'].index(end_date)   
        self.time_span = time_span

    def gen_data_baseline(self):
        price = self.price[:, self.stock_idx]
        feature = []
        rise_percent = []
        for idx in range(self.s_idx*self.minute, (self.e_idx+1)*self.minute, self.time_span):
            rise_percent.append((price[idx+self.time_span, :]-price[idx, :])/price[idx, :])
            feature.append(price[idx-self.time_span:idx, :])
        # feature[time*time_span*stock_num], rise_percent[time*stock_num]
        return np.array(feature), np.array(rise_percent)

    def gen_data_RL(self, feature_dim):
        price = self.price[:, self.stock_idx]
        feature = []
        rise_percent = []
        for idx in range(self.s_idx*self.minute, (self.e_idx+1)*self.minute, self.time_span):
            rise_percent.append((price[idx+self.time_span, :]-price[idx, :])/price[idx, :])
            feature_slice = np.zeros((len(self.stock_idx), feature_dim, 3))
            for t in range(feature_dim):
                price_slice = price[idx-(t+1)*self.time_span+1:idx-t*self.time_span+1, :]
                price_max = price_slice.max(axis=0)
                price_min = price_slice.min(axis=0)
                price_last = price_slice[-1, :]

                feature_slice[:, feature_dim-1-t, 0] = price_last / price[idx, :]
                feature_slice[:, feature_dim-1-t, 1] = price_max / price[idx, :]
                feature_slice[:, feature_dim-1-t, 2] = price_min / price[idx, :]
            feature.append(feature_slice)
        # feature[time*stock_num*feature_dim*3], rise_percent[time*stock_num]
        return np.array(feature), np.array(rise_percent), price[:self.s_idx*self.minute]

