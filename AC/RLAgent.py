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
        self.DM.input_range(DATE[0][0], DATE[0][1], self.config['stocks'])
        train_fea, train_rp, train_p = self.DM.gen_data()
        self.DM.input_range(DATE[0][1], DATE[0][2], self.config['stocks'])
        test_fea, test_rp, test_p = self.DM.gen_data()
        self.PVM = np.random.rand(*train_rp.shape)

        init_op = tf.global_variables_initializer()

        result_list = []

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            self.RL.assign_lr(sess, self.config['lr'])
            for episode in range(EPISODES):
                R = 0
                print("epoch: {}".format(episode))
                for epoch in range(EPOCHS):
                    idx = np.random.randint(train_p.shape[0]-batch_f)
                    epo_fea = train_fea[idx:idx+batch_feature]
                    epo_rp = train_rp[idx:idx+batch_f]
                    if idx == 0:
                        prev = min_max_norm(np.random.rand(train_p.shape[1]))
                    else:
                        prev = min_max_norm(self.PVM[idx-1])
                    f, r = self.RL.run_epoch(sess, epo_fea, epo_rp, self.RL.adam_op, prev)
                    R += np.sum(r)
                    self.PVM[idx:idx+batch_f] = f
                test_f, test_r = self.RL_test(sess, test_fea, test_rp)
                result = self.evaluation(test_p, test_f, test_p.shape[0]/240)
                # result_list.append(result)
                print(result)

    def RL_test(self, sess, feature, rise_percent):
        prev = np.ones(rise_percent.shape[1]) / rise_percent.shape[1]
        time_step = feature.shape[0]
        batch_num = (time_step - batch_prev) // batch_f
        padding_num = (time_step - batch_prev) % batch_f
        total_reward = []
        total_f = []
        if padding_num == 0:
            for iter in range(batch_num):
                test_fea = feature[iter*batch_f:(iter+1)*batch_f+batch_prev]
                test_rp = rise_percent[iter*batch_f:(iter+1)*batch_f]
                
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
                test_fea = feature[iter*batch_f:(iter+1)*batch_f+batch_prev]
                test_rp = rise_percent[iter*batch_f+batch_prev:(iter+1)*batch_f+batch_prev]
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





