import tensorflow as tf
import numpy as np

from model_simple import RRL
from Data import DM

class BaselineAgent(object):
    def __init__(self, config):
        self.DM = DM()
        self.batch_size = config['batch_f']
        self.config = config

        self.DM.input_range(config['start_date'], config['end_date'], config['time_span'], config['stocks'])
        self.feature, self.rise_percent, self.price = self.DM.gen_data_baseline()
        self.p_train = config['p_train']
        self.p_vali = config['p_vali']
        self.t_num = self.feature.shape[0]
        self.t_num_train = int(self.t_num * self.p_train)
        self.t_num_vali = int(self.t_num * self.p_vali)
        self.t_num_test = self.t_num - self.t_num_train - self.t_num_vali
        self.s_num = self.rise_percent.shape[1]    
        

    def evaluation(self, price, portfolio):
        money_sequence = []
        money = 100
        own = np.zeros(price.shape[1])
        for time_step in range(price.shape[0]):
            money = money + np.sum(own * price[time_step, :])
            money_sequence.append(money)
            own_new = money * portfolio[time_step, :] / price[time_step, :]
            fee = self.config["cost"] * np.sum(np.abs(own_new - own) * price[time_step, :])
            money = money - fee
            own = money * portfolio[time_step, :] / price[time_step, :]
            money = 0
        money_sequence.append(money + np.sum(own * price[time_step, :]))
        print(self.metrics(np.array(money_sequence)))
        return    

    def metrics(self, seq):
        length = len(seq)
        ret = seq[1:] - seq[:-1]
        ret_rate = (seq[1:] - seq[:-1]) / seq[:-1]

        ar = seq[-1] / seq[0]
        ar = np.power(ar, 240 / length) - 1
        sr = ret_rate.mean() / ret_rate.std()
        vol = ret.std()
        ir = ret.mean() / ret.std()
        md = -ret_rate.min()

        return [round(ar, 4), round(sr, 4), round(vol, 4), round(ir, 4), round(md, 4)]

    def best_stock(self):
        train_fea = self.feature[:self.t_num_train]
        train_rp = self.rise_percent[:self.t_num_train] 
        train_p = self.price[:self.t_num_train]   

        vali_fea = self.feature[self.t_num_train:self.t_num_train+self.t_num_vali]
        vali_rp = self.rise_percent[self.t_num_train:self.t_num_train+self.t_num_vali]    
        vali_p = self.price[self.t_num_train:self.t_num_train+self.t_num_vali] 

        test_fea = self.feature[self.t_num_train+self.t_num_vali:]
        test_rp = self.rise_percent[self.t_num_train+self.t_num_vali:]    
        test_p = self.price[self.t_num_train+self.t_num_vali:]         

        portfolios = np.zeros(test_rp.shape)
        time_step = portfolios.shape[0]
        stock_num = portfolios.shape[1]

        for t in range(time_step):
            fea = test_fea[t]
            pre_rp = (fea[-1, :] - fea[0]) / fea[0]
            select_stock = np.argmax(pre_rp)
            portfolios[t][select_stock] = 1
        self.evaluation(test_p, portfolios)
        
        return 

    def UBAH(self):
        train_fea = self.feature[:self.t_num_train]
        train_rp = self.rise_percent[:self.t_num_train] 
        train_p = self.price[:self.t_num_train]   

        vali_fea = self.feature[self.t_num_train:self.t_num_train+self.t_num_vali]
        vali_rp = self.rise_percent[self.t_num_train:self.t_num_train+self.t_num_vali]    
        vali_p = self.price[self.t_num_train:self.t_num_train+self.t_num_vali] 

        test_fea = self.feature[self.t_num_train+self.t_num_vali:]
        test_rp = self.rise_percent[self.t_num_train+self.t_num_vali:]    
        test_p = self.price[self.t_num_train+self.t_num_vali:]  

        portfolios = np.zeros(test_rp.shape)

        time_step = portfolios.shape[0]
        stock_num = portfolios.shape[1]       

        portfolios[0] = np.ones(stock_num) / stock_num
        for t in range(1, time_step):
            new_f = portfolios[t-1]*(test_rp[t-1]+1)
            new_f = new_f / np.sum(new_f)
            portfolios[t] = new_f
        self.evaluation(test_p, portfolios)
        
        return        

    def UCRP(self):
        train_fea = self.feature[:self.t_num_train]
        train_rp = self.rise_percent[:self.t_num_train] 
        train_p = self.price[:self.t_num_train]   

        vali_fea = self.feature[self.t_num_train:self.t_num_train+self.t_num_vali]
        vali_rp = self.rise_percent[self.t_num_train:self.t_num_train+self.t_num_vali]    
        vali_p = self.price[self.t_num_train:self.t_num_train+self.t_num_vali] 

        test_fea = self.feature[self.t_num_train+self.t_num_vali:]
        test_rp = self.rise_percent[self.t_num_train+self.t_num_vali:]    
        test_p = self.price[self.t_num_train+self.t_num_vali:]  

        portfolios = np.zeros(test_rp.shape)

        time_step = portfolios.shape[0]
        stock_num = portfolios.shape[1]

        portfolios = np.ones(portfolios.shape) / stock_num

        self.evaluation(test_p, portfolios)
        
        return  