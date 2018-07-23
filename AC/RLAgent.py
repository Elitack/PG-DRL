import tensorflow as tf
import numpy as np

from model_Attention import RRL
from data import DM
from tool import *


class RLAgent(object):
    def __init__(self, config):
        self.DM = DM()
        self.config = config
        self.RL = RRL(config)

    def RL_train(self):
        date = self.config['date']
        self.DM.input_range(date+'0101', date+'0701', self.config['stocks'])
        train_fea, train_rp, train_p = self.DM.gen_data()

        self.DM.input_range(date+'0701', date+'1001', self.config['stocks'])
        vali_fea, vali_rp, vali_p = self.DM.gen_data()

        self.DM.input_range(date+'1001', str(int(date)+1)+'0101', self.config['stocks'])
        test_fea, test_rp, test_p = self.DM.gen_data()

        self.PVM = np.ones((train_p.shape[0], train_p.shape[1])) / train_p.shape[1]
        train_idx = np.arange(train_p.shape[0]-batch_f+1)
        np.random.shuffle(train_idx)

        init_op = tf.global_variables_initializer()

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(init_op)
        self.RL.assign_lr(self.sess, self.config['lr'])
        for episode in range(EPISODES):
            R = 0
            print("epoch: {}".format(episode))
            for idx in train_idx:
                epo_fea = train_fea[idx:idx+batch_feature]
                epo_rp = train_rp[idx:idx+batch_f]
                if idx == 0:
                    prev = np.ones(train_p.shape[1]) / train_p.shape[1]
                else:
                    prev = self.PVM[idx-1]
                f, r = self.RL.run_epoch(self.sess, epo_fea, epo_rp, self.RL.adam_op, prev)
                R += np.sum(r)
                self.PVM[idx:idx+batch_f] = f

            print('vali:')
            self.vali_f, vali_r = self.RL_test(self.sess, vali_fea, vali_rp)
            result = self.evaluation(vali_p, self.vali_f, vali_p.shape[0])
            print('test')                
            self.test_f, test_r = self.RL_test(self.sess, test_fea, test_rp)
            result = self.evaluation(test_p, self.test_f, test_p.shape[0])
            print('\n\n')

    def RL_test(self, sess, feature, rise_percent):
        prev = np.ones(rise_percent.shape[1]) / rise_percent.shape[1]
        time_step = feature.shape[0]
        batch_num = (time_step - batch_prev) // batch_f
        padding_num = (time_step - batch_prev) % batch_f
        total_reward = []
        total_f = []
        if padding_num == 0:
            for itr in range(batch_num):
                test_fea = feature[itr*batch_f:(itr+1)*batch_f+batch_prev]
                test_rp = rise_percent[itr*batch_f:(itr+1)*batch_f]
                
                test_f, test_r = self.RL.run_test_epoch(sess, test_fea, test_rp, prev)

                prev = test_f[-1]
                total_reward.extend(list(test_r))
                total_f.append(test_f)
            total_f = np.array(total_f)
            total_f = total_f.reshape(-1, total_f.shape[2])
        else:
            padding_num = batch_f - padding_num
            feature = np.concatenate((feature, np.zeros((padding_num, feature.shape[1], feature.shape[2], feature.shape[3]))), axis=0)
            rise_percent = np.concatenate((rise_percent, np.zeros((padding_num, rise_percent.shape[1]))), axis=0)
            for itr in range(batch_num+1):
                test_fea = feature[itr*batch_f:(itr+1)*batch_f+batch_prev]
                test_rp = rise_percent[itr*batch_f:(itr+1)*batch_f]
                test_f, test_r = self.RL.run_test_epoch(sess, test_fea, test_rp, prev)

                prev = test_f[-1]
                total_reward.extend(list(test_r))
                total_f.append(test_f)
            total_reward = total_reward[:-padding_num]
            total_f = np.array(total_f)
            total_f = total_f.reshape(-1, total_f.shape[2])
            total_f = total_f[:-padding_num]
        return total_f, total_reward

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
        return self.metrics(np.array(money_sequence), days)

    def metrics(self, seq, days):
        ret = seq[1:] - seq[:-1]
        ret_rate = (seq[1:] - seq[:-1]) / seq[:-1]

        ar = seq[-1] / seq[0]
        ar = np.power(ar, 240 / days) - 1
        sr = ret_rate.mean() / ret_rate.std()
        vol = ret.std()
        ir = ret.mean() / ret.std()
        md = -ret_rate.min()

        return [round(ar, 4), round(sr, 4), round(vol, 4), round(ir, 4), round(md, 4)]





