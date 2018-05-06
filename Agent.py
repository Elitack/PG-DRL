import tensorflow as tf
import numpy as np

from model_simple import RRL
from Data import DM

class Agent(object):
    def __init__(self, config):
        self.DM = DM()
        self.batch_size = config['batch_size']
        self.config = config

        self.DM.input_range(config['start_date'], config['end_date'], config['time_span'], config['stocks'])

        self.p_train = config['p_train']
        self.p_vali = config['p_vali']
        self.t_num = self.feature.shape[0]
        self.t_num_train = int(self.t_num * self.p_train)
        self.t_num_vali = int(self.t_num * self.p_vali)
        self.t_num_test = self.t_num - self.t_num_train - self.t_num_vali
        self.s_num = self.rise_percent.shape[1]        




    def train(self):
        pass

    def test(self):
        pass




