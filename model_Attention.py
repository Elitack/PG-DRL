import tensorflow as tf
import numpy as np

from model_simple import RRL
from Data import DM
from Agent import Agent

class RLAgent(Agent):
    def __init__(self, config):
        super().__init__(config)
        self.RL = RRL(config)
        self.feature, self.rise_percent = self.DM.gen_data_RL(self.config["fea_dim"])

    def RL_train(self, epochs=100):
        vali_fea = self.feature[self.t_num_train:self.t_num_train+self.t_num_vali]
        vali_rp = self.rise_percent[self.t_num_train:self.t_num_train+self.t_num_vali]          
        test_fea = self.feature[self.t_num_train+self.t_num_vali:]
        test_rp = self.rise_percent[self.t_num_train+self.t_num_vali:]        

        self.PVM = np.ones((self.t_num_train, self.s_num)) / self.s_num

        train_idx = np.arange(self.t_num_train-self.batch_size)
        np.random.shuffle(train_idx)
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            self.RL.assign_lr(sess, 0.01)
            for epo in range(epochs):
                R = 0
                print("epoch: {}".format(epo))
                for idx in train_idx:
                    epo_fea = self.feature[idx:idx+self.batch_size]
                    epo_rp = self.rise_percent[idx:idx+self.batch_size]
                    if idx == 0:
                        prev = np.ones(self.rise_percent.shape[1]) / self.rise_percent.shape[1]
                    else:
                        prev = self.PVM[idx-1]
                    f, r = self.RL.run_epoch(sess, epo_fea, epo_rp, self.RL.adam_op, prev)
                    R += np.sum(r)
                    self.PVM[idx:idx+self.batch_size] = f
                print(R / self.t_num_train)

                test_f, test_r = self.RL_test(sess, test_fea, test_rp)
                print(np.sum(test_r))

    def RL_test(self, sess, feature, rise_percent):
        prev = np.ones(rise_percent.shape[1]) / rise_percent.shape[1]
        time_step = feature.shape[0]
        batch_num = time_step // self.batch_size
        padding_num = feature.shape[0] % self.batch_size
        total_reward = []
        total_f = []
        if padding_num == 0:
            for iter in range(batch_num):
                test_fea = feature[iter*self.batch_size:(iter+1)*self.batch_size]
                test_rp = rise_percent[iter*self.batch_size:(iter+1)*self.batch_size]
                test_f, test_r = self.RL.run_test_epoch(sess, test_fea, test_rp, prev)

                prev = test_f[-1]
                total_reward.extend(list(test_r))
                total_f.append(test_f)
            total_f = np.array(total_f)
            total_f = total_f.reshape(-1, total_f.shape[2])
        else:

            feature = np.concatenate((feature, np.zeros((padding_num, feature.shape[1], feature.shape[2], feature.shape[3]))), axis=0)
            rise_percent = np.concatenate((rise_percent, np.zeros((padding_num, rise_percent.shape[1]))), axis=0)

            for iter in range(batch_num+1):
                test_fea = feature[iter*self.batch_size:(iter+1)*self.batch_size]
                test_rp = rise_percent[iter*self.batch_size:(iter+1)*self.batch_size]
                test_f, test_r = self.RL.run_test_epoch(sess, test_fea, test_rp, prev)

                prev = test_f[-1]
                total_reward.extend(list(test_r))
                total_f.append(test_f)
            total_reward = total_reward[:-padding_num]
            total_f = np.array(total_f)
            total_f = total_f.reshape(-1, total_f.shape[2])
            total_f = total_f[:-padding_num]
        return total_f, total_reward







