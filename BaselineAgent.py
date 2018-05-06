import tensorflow as tf
import numpy as np

from model_simple import RRL
from Data import DM
from Agent import Agent

class BaselineAgent(Agent):
    def __init__(self, config):
        super().__init__(config)
        self.feature, self.rise_percent = self.DM.gen_data_baseline(self.config["fea_dim"])

    def best_stock(self):
        test_fea = self.feature[self.t_num_train+self.t_num_vali:]
        test_rp = self.rise_percent[self.t_num_train+self.t_num_vali:]

        portfolios = np.zeros(test_rp.shape)
        time_step = portfolios.shape[0]
        stock_num = portfolios.shape[1]

        for t in range(time_step):
            fea = test_fea[t]
            pre_rp = (fea[-1, :] - fea[0]) / fea[0]
            select_stock = np.argmax(pre_rp)
            portfolios[t][select_stock] = 1
        return portfolios

    def UBAH(self):
        test_rp = self.rise_percent[self.t_num_train+self.t_num_vali:]
        portfolios = np.zeros(test_rp.shape)

        time_step = portfolios.shape[0]
        stock_num = portfolios.shape[1]       

        portfolios[0] = np.ones(stock_num) / stock_num
        for t in range(1, time_step):
            new_f = portfolios[t-1]*(test_rp[t-1]+1)
            new_f = new_f / np.sum(new_f)
            portfolios[t] = new_f
        return portfolios        

    def UCRP(self):
        test_rp = self.rise_percent[self.t_num_train+self.t_num_vali:]
        portfolios = np.zeros(test_rp.shape)

        time_step = portfolios.shape[0]
        stock_num = portfolios.shape[1]

        portfolios = np.ones(portfolios.shape) / stock_num

        return portfolios