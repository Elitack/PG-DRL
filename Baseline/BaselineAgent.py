import tensorflow as tf
import numpy as np
from tool import *
from data import DM

class BaselineAgent(object):
    def __init__(self, config):
        self.DM = DM()
        self.config = config
        self.DM.input_range(DATE[1][0], DATE[1][1], self.config['stocks'])
        self.train_fea, self.train_rp, self.train_p = self.DM.gen_data()
        self.DM.input_range(DATE[1][1], DATE[1][2], self.config['stocks'])
        self.test_fea, self.test_rp, self.test_p = self.DM.gen_data()        

    def evaluation(self, price, portfolio, days):
        money_sequence = []
        money = 100
        own = np.zeros(price.shape[1])
        for time_step in range(price.shape[0]):
            money = money + np.sum(own * price[time_step, :])
            money_sequence.append(money)
            own_new = money * portfolio[time_step, :] / price[time_step, :]
            fee = cost * np.sum(np.abs(own_new - own) * price[time_step, :])
            money = money - fee
            own = money * portfolio[time_step, :] / price[time_step, :]
            money = 0
        money_sequence.append(money + np.sum(own * price[time_step, :]))
        print(self.metrics(np.array(money_sequence), days))
        return    

    def metrics(self, seq, days):
        length = len(seq)
        ret = seq[1:] - seq[:-1]
        ret_rate = (seq[1:] - seq[:-1]) / seq[:-1]

        ar = seq[-1] / seq[0]
        ar = np.power(ar, 240 / days) - 1
        sr = ret_rate.mean() / ret_rate.std()
        vol = ret.std()
        ir = ret.mean() / ret.std()
        md = -ret_rate.min()

        return [round(ar, 4), round(sr, 4), round(vol, 4), round(ir, 4), round(md, 4)]

    def best_stock(self):
        portfolios = np.zeros(self.test_rp.shape)
        time_step = portfolios.shape[0]
        stock_num = portfolios.shape[1]

        for t in range(time_step):
            fea = self.test_fea[t]
            pre_rp = (fea[:, -1] - fea[:, 0]) / fea[:, 0]
            select_stock = np.argmax(pre_rp)
            portfolios[t][select_stock] = 1
        self.evaluation(self.test_p, portfolios, time_step)
        
        return 

    def UBAH(self):
        portfolios = np.zeros(self.test_rp.shape)
        time_step = portfolios.shape[0]
        stock_num = portfolios.shape[1]      

        portfolios[0] = np.ones(stock_num) / stock_num
        for t in range(1, time_step):
            new_f = portfolios[t-1]*(self.test_rp[t-1]+1)
            new_f = new_f / np.sum(new_f)
            portfolios[t] = new_f
        self.evaluation(self.test_p, portfolios, time_step)
        
        return        

    def UCRP(self):
        portfolios = np.zeros(self.test_rp.shape)
        time_step = portfolios.shape[0]
        stock_num = portfolios.shape[1]

        portfolios = np.ones(portfolios.shape) / stock_num

        self.evaluation(self.test_p, portfolios, time_step)
        
        return  

    def eg(self):
        b = np.ones(x.size)