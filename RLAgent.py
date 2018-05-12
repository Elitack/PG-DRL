import tensorflow as tf
import numpy as np

from model_simple import RRL
from Data import DM

class RLAgent(object):
    def __init__(self, config):
        self.DM = DM()
        self.batch_feature = config['batch_feature']
        self.batch_f = config['batch_f']
        self.batch_prev = self.batch_feature - self.batch_f
        self.config = config

        self.DM.input_range(config['start_date'], config['end_date'], config['time_span'], config['stocks'])
        self.feature, self.rise_percent, self.price = self.DM.gen_data_RL(self.config["fea_dim"], prev=self.batch_prev)

        self.p_train = config['p_train']
        self.p_vali = config['p_vali']
        self.t_num = self.feature.shape[0]-self.batch_prev

        self.t_num_train = int(self.t_num * self.p_train)
        self.t_num_vali = int(self.t_num * self.p_vali)
        self.t_num_test = self.t_num - self.t_num_train - self.t_num_vali
        self.s_num = self.rise_percent.shape[1]       

        self.RL = RRL(config)
        

    def RL_train(self, epochs=100):
        train_fea = self.feature[:self.t_num_train+self.batch_prev]
        train_rp = self.rise_percent[self.batch_prev:self.t_num_train+self.batch_prev] 
        train_p = self.price[self.batch_prev:self.t_num_train+self.batch_prev]   

        vali_fea = self.feature[self.t_num_train:self.t_num_train+self.t_num_vali+self.batch_prev]
        vali_rp = self.rise_percent[self.t_num_train+self.batch_prev:self.t_num_train+self.t_num_vali+self.batch_prev]    
        vali_p = self.price[self.t_num_train+self.batch_prev:self.t_num_train+self.t_num_vali+self.batch_prev] 

        test_fea = self.feature[self.t_num_train+self.t_num_vali:]
        test_rp = self.rise_percent[self.t_num_train+self.t_num_vali+self.batch_prev:]    
        test_p = self.price[self.t_num_train+self.t_num_vali+self.batch_prev:] 

        self.PVM = np.ones((self.t_num_train, self.s_num)) / self.s_num

        train_idx = np.arange(self.t_num_train+self.batch_prev-self.batch_feature)
        np.random.shuffle(train_idx)
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            self.RL.assign_lr(sess, self.config['lr'])
            for epo in range(epochs):
                R = 0
                print("epoch: {}".format(epo))
                for idx in train_idx:
                    epo_fea = train_fea[idx:idx+self.batch_feature]
                    epo_rp = train_rp[idx:idx+self.batch_f]
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
                self.evaluation(test_p, test_f)

    def RL_test(self, sess, feature, rise_percent):
        prev = np.ones(rise_percent.shape[1]) / rise_percent.shape[1]
        time_step = feature.shape[0]
        batch_num = (time_step - self.batch_prev) // self.batch_f
        padding_num = (time_step - self.batch_prev) % self.batch_size
        total_reward = []
        total_f = []
        if padding_num == 0:
            for iter in range(batch_num):
                test_fea = feature[iter*self.batch_f:(iter+1)*self.batch_f+self.batch_prev]
                test_rp = rise_percent[iter*self.batch_f+self.batch_prev:(iter+1)*self.batch_f+self.batch_prev]
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
                test_fea = feature[iter*self.batch_f:(iter+1)*self.batch_f+self.batch_prev]
                test_rp = rise_percent[iter*self.batch_f+self.batch_prev:(iter+1)*self.batch_f+self.batch_prev]
                test_f, test_r = self.RL.run_test_epoch(sess, test_fea, test_rp, prev)

                prev = test_f[-1]
                total_reward.extend(list(test_r))
                total_f.append(test_f)
            total_reward = total_reward[:-padding_num]
            total_f = np.array(total_f)
            total_f = total_f.reshape(-1, total_f.shape[2])
            total_f = total_f[:-padding_num]
        return total_f, total_reward

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
        print(self.metrics(money_sequence))
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





