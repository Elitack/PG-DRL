import numpy as np
from data import DM
import random
from tool import *

class Env(object):
    def __init__(self, config):
        self.DM = DM()
        self.DM.input_range(config['start_date'], config['end_date'], config['stocks'])
        self.fea, self.rp, self.p = self.DM.gen_data(config["fea_dim"])
        self.count = 0
        self.time_num = self.rp.shape[0]
        self.stock_num = self.rp.shape[1]
        self.reset()

    def reset(self):
        self.PVM = np.random.rand(self.rp.shape)

    def select_batch(self):
        self.idx = np.random.randint(batch_prev, int(self.time_num * tv_s) - batch_f)
        return self.fea[self.idx-batch_prev:self.idx+1], self.PVM[self.idx-1]

    def step(self, action):
        next_state = self.fea[self.idx+1]
        reward = np.dot(action, self.rp[self.idx]) - cost * np.sum(np.abs(action - self.PVM[self.idx - 1]))
        self.PVM[self.idx] = action
        self.idx = self.idx + 1
        return next_state, reward
