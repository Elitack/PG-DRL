import tensorflow as tf
import numpy as np

class RRL(object):
    
    ''' Model Parameters '''
    def _weight_variable(self, shape, name=""):
        return tf.get_variable(name, shape, tf.float32)

    def _bias_variable(self, shape, name=""):
        return tf.get_variable(name, shape, tf.float32)

    ''' Stock Scoring Module '''
    def _cnn_net(self, state_vec, prev, name=""):
        '''
        A linear model for generating score of each stock, by linearly combining their feature vectors.
        :param state_vec: stock_num * feature dimension
        :param name: optional name
        :return: stock score with size [stock_num * 1]
        '''
        shape = state_vec.get_shape().as_list()
        conv1 = tf.nn.relu(\
            tf.nn.conv2d(state_vec, self._weight_variable([1, 3, 3, 2], 'conv1_w'), [1, 1, 1, 1], "VALID")\
            + self._bias_variable([2], 'conv1_b'))
        conv2 = tf.nn.relu(\
            tf.nn.conv2d(conv1, self._weight_variable([1, self.config["fea_dim"]-2, 2, 20], 'conv2_w'), [1, 1, 1, 1], "VALID")\
            + self._bias_variable([20], 'conv2_b'))
        concat_w = tf.concat([conv2, prev], 3)
        conv3 = tf.squeeze(\
            tf.nn.conv2d(concat_w, self._weight_variable([1, 1, 21, 1], 'conv3_w'), [1, 1, 1, 1], "VALID"))
        return conv3

    def _score2f(self, score):
 
        return tf.nn.softmax(score)

    def __init__(self, config):
        self.config = config
        self.state_vec = state_vec = \
                tf.placeholder(dtype=tf.float32, shape=[config['batch_size'], config['stock_num'], config["fea_dim"], 3], name="state_vec") 
        self.Fp = Fp = \
                tf.placeholder(dtype=tf.float32, shape=[1, config["stock_num"], 1, 1], name="Portfolio_previous")
        self.rise_percent = rise_percent = \
                tf.placeholder(dtype=tf.float32, shape=[config['batch_size'], config["stock_num"]], name="rise_percent")
        
        self.lr = lr = \
                tf.Variable(0.0, trainable=False, name="Learning_Rate")
        list_F = []
        list_reward = []

        batch = config['batch_size']

        cat_w = self._weight_variable([config["stock_num"], config["stock_num"]], "cat2policy_w")
        cat_b = self._bias_variable(config["stock_num"], "cat2policy_b")        

        for item in range(batch):
            state_vec_t = tf.slice(state_vec, [0, 0, 0, 0], [1, -1, -1, -1])
            rise_percent_t = tf.squeeze(tf.slice(rise_percent, [0, 0], [1, -1]))
            if item == 0:
                with tf.variable_scope("model"):
                    score = self._cnn_net(state_vec_t, Fp, 'CNN')
                    score_cat = tf.matmul(tf.expand_dims(score, 0), cat_w) + cat_b   
                    F = self._score2f(score_cat)

                # RL reward
                Rt = tf.reduce_sum(tf.multiply(F, rise_percent)) - \
                        config["cost"] * tf.reduce_sum(tf.abs(F-Fp)) # config["cost"] = 0.003, turnover cost

                list_F.append(F)
                list_reward.append(Rt)
            else:
                with tf.variable_scope("model", reuse=True):
                    score = self._cnn_net(state_vec_t, tf.reshape(F, [1, -1, 1, 1]), 'CNN')
                    score_cat = tf.matmul(tf.expand_dims(score, 0), cat_w) + cat_b
                    F = self._score2f(score_cat)

                # RL reward
                Rt = tf.reduce_sum(tf.multiply(F, rise_percent)) - \
                        config["cost"] * tf.reduce_sum(tf.abs(F-Fp)) # config["cost"] = 0.003, turnover cost   

                list_F.append(F)
                list_reward.append(Rt)

        self.F = tf.stack(list_F)
        self.reward = tf.stack(list_reward)
        # RL optimization part
        adam_optimizer = tf.train.AdamOptimizer(learning_rate = lr)
        self.adam_op = adam_optimizer.minimize(-tf.reduce_sum(self.reward))

        # Subgraph for learning rate adaptation
        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_lr")
        self._lr_update = tf.assign(self.lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})


    def run_step(self, session, state_vec, rise_percent, Fp, op):
        ''' 
        Stepwise Trading
        :param session: tensorflow session
        :param state_vec: state vector on stocks in trading days
        :param rise_percent: rise_percent of all stocks in trading days
        :param op: optimization option
        :return: portfolios, sharpe ratio, variable of policy net
        '''
        feed_dict = {}
        feed_dict[self.Fp] = Fp
        feed_dict[self.state_vec] = state_vec
        feed_dict[self.rise_percent] = rise_percent

        # Do optimization
        fetches = [self.F, self.reward, op]
        portfolio, reward, _ = session.run(fetches, feed_dict)
        return portfolio, reward
    
    def run_epoch(self, session, state_vec, rise_percent, op, Prev):
        '''
        Epoch-wise Runing
        :param session: tensorflow session
        :param state_vec: state vec of all stocks in trading days
        :param rise_percent: rise_percent of all stocks in trading days
        :param op: optimization option
        :return: Sharpe Ratio, Portfolios
        '''
        times, stock_num, fea_dim, channel = state_vec.shape
        total_reward = 0
        portfolios, reward = \
                self.run_step(session, state_vec, rise_percent, Prev.reshape(1, -1, 1, 1), op)
        return portfolios, reward

    def run_test_epoch(self, session, state_vec, rise_percent, Prev):
        '''
        For test
        '''
        times, stock_num, fea_dim, channel = state_vec.shape
        feed_dict = {}
        feed_dict[self.Fp] = Prev.reshape(1, -1, 1, 1)
        feed_dict[self.state_vec] = state_vec
        feed_dict[self.rise_percent] = rise_percent
        fetches = [self.F, self.reward]
        portfolios, reward = session.run(fetches, feed_dict)
        return portfolios, reward
