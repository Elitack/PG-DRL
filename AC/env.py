import numpy as np
from data import DM


class Env(object):
    def __init__(self, config):
        self.DM = DM()
        self.batch_size = config['batch_size']
        self.DM.input_range(config['start_date'], config['end_date'], config['stocks'])
        self.fea, self.rp, self.p = self.DM.gen_data(self.config["fea_dim"])   
        self.count = 0
        self.time_num = self.rp.shape[0]
        self.stock_num = self.rp.shape[1]
        self.PVM = np.ones(self.rp.shape)

    def reset():
        self.count = 0
        return self.fea[count]