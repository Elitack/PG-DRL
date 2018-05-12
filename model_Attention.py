import tensorflow as tf
import numpy as np

class RRL(object):
    
    ''' Model Parameters '''
    def _weight_variable(self, shape, name=""):
        return tf.get_variable(name, shape, tf.float32)

    def _bias_variable(self, shape, name=""):
        return tf.get_variable(name, shape, tf.float32)

    ''' Stock Scoring Module '''
    def _cnn_net(self, state_vec, name=""):
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
        conv2 = tf.squeeze(\
            tf.nn.conv2d(conv1, self._weight_variable([1, self.config["fea_dim"]-2, 2, 1], 'conv2_w'), [1, 1, 1, 1], "VALID"))
        return conv2

    def _score2f(self, score, axis=-1):
 
        return tf.nn.softmax(score, axis=axis)

    def __init__(self, config):
        self.config = config

        batch_feature = config['batch_feature']
        batch_f = config['batch_f']
        batch_prev = self.batch_feature - self.batch_f

        self.state_vec = state_vec = \
                tf.placeholder(dtype=tf.float32, shape=[batch_feature, config['stock_num'], config["fea_dim"], 3], name="state_vec") 
        self.Fp = Fp = \
                tf.placeholder(dtype=tf.float32, shape=[config["stock_num"], 1], name="Portfolio_previous")
        self.rise_percent = rise_percent = \
                tf.placeholder(dtype=tf.float32, shape=[batch_f, config["stock_num"]], name="rise_percent")
        
        self.lr = lr = \
                tf.Variable(0.0, trainable=False, name="Learning_Rate")



        cat_w = self._weight_variable([config["stock_num"], config["stock_num"] * 2], "cat2policy_w")
        cat_b = self._bias_variable([config["stock_num"], 1], "cat2policy_b")   

        p2o_w = self._weight_variable([config["stock_num"], 1], "p2o_w")
        p2o_b = self._bias_variable([config["stock_num"], 1], "p2o_b")     

        attention_layer = self._weight_variable([batch_prev+1, 1], "attention")

        score = self._cnn_net(state_vec, 'CNN')
        score = tf.expand_dims(self._score2f(score), 0)
        lstm_cell = rnn.BasicLSTMCell(num_units=config["stock_num"], forget_bias=1.0, state_is_tuple=True)
        init_state = lstm_cell.zero_state(1, dtype=tf.float32)
        unprocessed_F = []
        with tf.variable_scope('RNN'):
            for timestep in range(batch_feature):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = lstm_cell(score[:, timestep, :], init_state)
                unprocessed_F.append(tf.squeeze(self._score2f(cell_output)))
        unprocessed_F = tf.stack(unprocessed_F)
        latent_v = tf.sigmoid(tf.matmul(F, p2o_w)+p2o_b)

        processed_F = []
        for item in range(batch_f):
            batch_v = tf.slice(latent_v, [item, 0], [batch_prev+1, -1])
            batch_f = tf.slice(unprocessed_F, [item, 0], [batch_prev+1, -1])
            ratio = tf.nn.softmax(tf.multiply(batch_v, attention_layer), axis=0)

            ele_f = tf.matmul(tf.transpose(batch_f), ratio)
            processed_F.append(ele_f)

        final_F = []
        list_reward = []
        for item in range(batch_f):
            rise_percent_t = tf.squeeze(tf.slice(rise_percent, [item, 0], [1, -1]))
            if item == 0:
                cat_layer = tf.concat([processed_F[item], Fp], axis=0)
                score_cat = tf.matmul(cat_w, cat_layer) + cat_b
                F = self._score2f(score_cat, 0)

                # RL reward
                Rt = tf.reduce_sum(tf.multiply(F, tf.transpose(rise_percent))) - \
                        config["cost"] * tf.reduce_sum(tf.abs(F-Fp)) # config["cost"] = 0.003, turnover cost

                final_F.append(F)
                list_reward.append(Rt)
            else:
                cat_layer = tf.concat([processed_F[item], F], axis=0)
                score_cat = tf.matmul(cat_w, cat_layer) + cat_b
                F = self._score2f(score_cat, 0)

                # RL reward
                Rt = tf.reduce_sum(tf.multiply(F, tf.transpose(rise_percent))) - \
                        config["cost"] * tf.reduce_sum(tf.abs(F-Fp)) # config["cost"] = 0.003, turnover cost

                list_F.append(F)
                list_reward.append(Rt)
        

        self.final_F = tf.stack(final_F)
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
                self.run_step(session, state_vec, rise_percent, Prev.reshape(-1, 1), op)
        return portfolios, reward

    def run_test_epoch(self, session, state_vec, rise_percent, Prev):
        '''
        For test
        '''
        times, stock_num, fea_dim, channel = state_vec.shape
        feed_dict = {}
        feed_dict[self.Fp] = Prev.reshape(-1, 1)
        feed_dict[self.state_vec] = state_vec
        feed_dict[self.rise_percent] = rise_percent
        fetches = [self.F, self.reward]
        portfolios, reward = session.run(fetches, feed_dict)
        return portfolios, reward
