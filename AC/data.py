from tool import *
import numpy as np
import json

data_dir = '../data/'
class DM(object):
    def __init__(self):   
        self.feature = np.load(data_dir+'SP500.npy')
        with open(data_dir+'tag.json', 'r') as f:
            self.stock_tag = json.load(f)  
        self.stock_num, self.date = self.feature[:, :, 0].shape
        self.feature = np.swapaxes(self.feature, 0, 1)

    def input_range(self, start_date, end_date, stocks='all'):
        if stocks == 'all':
            self.stock_idx = np.arange(self.stock_num)
        else:
            self.stock_idx = np.array([self.stock_tag['stock'].index(stock) for stock in stocks])  
        a_date = np.asarray(self.stock_tag['date'])
        self.s_idx = np.where(a_date >= start_date)[0][0]
        self.e_idx = np.where(a_date >= end_date)[0][0]
        print('start_date:{}'.format(a_date[self.s_idx]))
        print('end_date:{}'.format(a_date[self.e_idx]))

    def gen_data(self):

        if self.s_idx - FEATURE_DIM - batch_prev < 0:
            raise Exception('data is not enough')

        price = self.feature[:, self.stock_idx, :]
        feature = []
        rise_percent = []
        price_need = []

        for idx in range(self.s_idx - batch_prev, self.e_idx):
            if idx >= self.s_idx:
                rise_percent.append((price[idx+1, :, 2]-price[idx, :, 2]) / price[idx, :, 2])
                price_need.append(price[idx, :])

            feature_slice = np.zeros((len(self.stock_idx), FEATURE_DIM, 3))
            for t in range(FEATURE_DIM):
                feature_slice[:, FEATURE_DIM-1-t, 0] = price[idx-t, :, 2] / price[idx, :, 2]
                feature_slice[:, FEATURE_DIM-1-t, 1] = price[idx-t, :, 1] / price[idx, :, 2]
                feature_slice[:, FEATURE_DIM-1-t, 2] = price[idx-t, :, 0] / price[idx, :, 2]
            feature.append(feature_slice)
        # feature[time*stock_num*feature_dim*3], rise_percent[time*stock_num]
        return np.array(feature), np.array(rise_percent), np.array(price_need)

