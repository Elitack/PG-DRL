import tensorflow as tf
import numpy as np
import json

data_dir = '../data/'
class DM(object):
    def __init__(self):   
        feature = np.load(data_dir+'processed_feature.npy')
        with open(data_dir+'processed_stock_tag.json', 'r') as f:
            self.stock_tag = json.load(f)  
        self.stock_num, self.date, self.minute = feature[:, :, :, 0].shape
        self.price = feature[:, :, :, 0].reshape(self.stock_num, -1).T
        self.volume = feature[:, :, :, 1].reshape(self.stock_num, -1).T

    def input_range(self, start_date, end_date, time_span, stocks='all'):
        if stocks == 'all':
            self.stock_idx = np.arange(self.stock_num)
        else:
            self.stock_idx = np.array([self.stock_tag['stock_map'].index(stock) for stock in stocks])  
        a_date = np.asarray(self.stock_tag['date_stamp'])
        self.s_idx = np.where(a_date >= start_date)[0][0]
        self.e_idx = np.where(a_date >= end_date)[0][0]
        print('start_date:{}'.format(a_date[self.s_idx]))
        print('end_date:{}'.format(a_date[self.e_idx]))
        self.time_span = time_span

    def gen_data(self, feature_dim):

        if self.s_idx * self.minute - feature_dim * self.time_span < 0:
            raise Exception('data is not enought')

        price = self.price[:, self.stock_idx]
        feature = []
        rise_percent = []
        price_need = []

        for idx in range(self.s_idx*self.minute, self.e_idx*self.minute, self.time_span):
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
            price_need.append(price[idx, :])
        # feature[time*stock_num*feature_dim*3], rise_percent[time*stock_num]

        return np.array(feature), np.array(rise_percent), np.array(price_need)

