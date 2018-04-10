import tensorflow as tf
import numpy as np

class RRL(object):
    
    ''' Model Parameters '''
    def _weight_variable(self, shape, name=""):
        return tf.get_variable(name, shape, tf.float32)

    def _bias_variable(self, shape, name=""):
        return tf.get_variable(name, shape, tf.float32)

    ''' Stock Scoring Module '''
    def _linear_net(self, state_vec, name=""):
        '''
        A linear model for generating score of each stock, by linearly combining their feature vectors.
        :param state_vec: stock_num * feature dimension
        :param name: optional name
        :return: stock score with size [stock_num * 1]
        '''
        shape = state_vec.get_shape().as_list()
        mean, variance = tf.nn.moments(state_vec, axes=[0], keep_dims=False)
        input = tf.nn.batch_normalization(state_vec, mean, variance, offset=0, scale=1, variance_epsilon=0.0001)
        logi_w = self._weight_variable(shape=[shape[1], 1], name="logi_w_"+name)
        logi_b = self._bias_variable([1], "logi_b_"+name)
        scores = tf.nn.sigmoid(tf.matmul(input, logi_w)+logi_b)
        return scores

    def _get_split_gradient(self, F, var_net):
        '''
        For computing the gradient for each element in var_net with regard to each element in F
        :param F: a vector, each Fi has a functional reliance to var_net
        :param var_net: trading variables in the network
        :return: a list with same length to var_net, each element in the list is the corresponding gradients
        '''
        grad_set = [[] for _ in range(len(var_net))]
        for i in range(F.get_shape().as_list()[1]):
            # split each Fi
            Fi = tf.slice(F, [0, i], [-1, 1])
            # compute the gradient related to Fi
            grad = tf.gradients(Fi, var_net)
            for j in range(len(var_net)):
                # save each grad in var_net
                grad_set[j].append(grad[j])
        # ret: var_net * [F_size * grad_dim]
        ret = []
        for i in range(len(grad_set)):
            ret.append(tf.stack(grad_set[i], axis=0))
        return ret

    def _compute_weighted_gradient(self, grad, weight):
        '''
        Compute the weighted gradient for each element in grad with regard to weight
        :param grad: grad is a list, each grad_i has the same size to weight
        :param weight: compute the weighted gradient for each grad_i
        :return: weighted gradient list, with the same size of grad
        '''
        ret = []
        for i in range(len(grad)):
            # for each grad_i
            shape = grad[i].get_shape().as_list()
            begin = [0] * len(shape)
            size = [-1] * len(shape)
            size[0] = 1
            # grad shape is F_size * grad_dim, now get the first grad
            slice_grad = tf.squeeze(tf.slice(grad[i], begin, size), axis=0)
            slice_weight = tf.squeeze(tf.slice(weight, [0, 0], [-1, 1]))
            # apply the first weight
            grad_i = slice_grad * slice_weight
            for j in range(1, len(grad)):
                # get next grad
                begin = [0] * len(shape)
                begin[0] = j
                slice_grad = tf.squeeze(tf.slice(grad[i], begin, size), axis=0)
                slice_weight = tf.squeeze(tf.slice(weight, [0, j], [-1, 1]))
                grad_i += slice_grad * slice_weight
            ret.append(grad_i)
        return ret

    def __init__(self, config):
        self.state_vec = state_vec = \
                tf.placeholder(dtype=tf.float32, shape=[config['stock_num'], config["fea_dim"]], name="state_vec") 
        self.Ap = Ap = \
                tf.placeholder(dtype=tf.float32, shape=[], name="A_previous")
        self.Bp = Bp = \
                tf.placeholder(dtype=tf.float32, shape=[], name="B_previous")
        self.Fp = Fp = \
                tf.placeholder(dtype=tf.float32, shape=[1, config["stock_num"]], name="Portfolio_previous")
        self.rise_percent = rise_percent = \
                tf.placeholder(dtype=tf.float32, shape=[config["stock_num"], 1], name="rise_percent")
        
        self.lr = lr = \
                tf.Variable(0.0, trainable=False, name="Learning_Rate")
        
        scores = self._linear_net(state_vec)

        ''' Reinforcement Learning Part '''
        cat_layer = tf.concat([tf.transpose(scores), Fp], axis=1)
        
        # Get the fc variables
        cat_w = self._weight_variable([config["stock_num"] * 2, config["stock_num"]], "cat2policy_w")
        cat_b = self._bias_variable(config["stock_num"], "cat2policy_b")
        # Get the new policy/portfolio currently
        score_cat = tf.matmul(cat_layer, cat_w) + cat_b
        self.F = F = \
                tf.nn.softmax(score_cat)

        var_net = tf.trainable_variables()
        # collect previous gradients
        grad_net_prev = []
        for i in range(len(var_net)):
            shape = [config["stock_num"]]
            shape.extend(var_net[i].get_shape().as_list())
            grad_net_prev.append(tf.placeholder(dtype=tf.float32, shape=shape))
        self.grad_prev = grad_net_prev

        # RL reward
        Rt = tf.matmul(F, rise_percent) - \
                config["cost"] * tf.reduce_sum(tf.abs(F-Fp)) # config["cost"] = 0.003, turnover cost

        # Update state
        self.A = config["gamma"] * Rt + (1-config["gamma"]) * Ap # config["gamma"] is exponential moving average param
        self.B = config["gamma"] * tf.square(Rt) + (1-config["gamma"]) * Bp
        K = tf.sqrt((1-config["gamma"]/2) / (1-config["gamma"]))
        self.sharpe = self.A / (K*(tf.sqrt(self.B-tf.square(self.A)))) # differential Sharpe Ratio
        # Save the gradient of F/net at current time, and it'll be feed as grad_prev next time
        self.grad_F_net_prev = self._get_split_gradient(F, var_net)
        grad_S_Rt = tf.gradients(self.sharpe, Rt) # S is the Reward, specifically the Sharpe Ratio here
        grad_Rt_F = tf.gradients(Rt, F, grad_ys = grad_S_Rt)
        grad_Rt_Fp = tf.gradients(Rt, Fp, grad_ys = grad_S_Rt)
        grad_F_net = tf.gradients(F, var_net, grad_ys = grad_Rt_F)
        grad_F_Fp = tf.gradients(F, Fp, grad_ys = grad_Rt_F)
        grad_F_net = [self._compute_weighted_gradient(self.grad_prev, grad_F_Fp[0])[i] + grad_F_net[i] \
                for i in range(len(grad_F_net))]
        grad_Fp_net = self._compute_weighted_gradient(self.grad_prev, grad_Rt_Fp[0])
        grad_net = []
        for i in range(len(grad_F_net)):
            grad_net.append(tf.add(grad_F_net[i], grad_Fp_net[i]) * -1)

        # RL optimization part
        adam_optimizer = tf.train.AdamOptimizer(learning_rate = lr)
        self.adam_op = adam_optimizer.apply_gradients(zip(grad_net, var_net))

        # Subgraph for learning rate adaptation
        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_lr")
        self._lr_update = tf.assign(self.lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def run_init(self, session, state_vec, rise_percent, Prev=None):
        '''
        Initialize the modell
        :param session: tf session
        :param state_vec: daily stock state, stock_num * feature_dimension
        :param rise_percent: daily stock rise_percent, stock_num * 1
        :param Prev: Given previous information as starting point
        :return: A, B, F, grad_prev[gradient of this step]
        '''
        feed_dict = {}
        feed_dict[self.Ap] = 0
        feed_dict[self.Bp] = 0
        feed_dict[self.feature] = feature
        feed_dict[self.rise_percent] = rise_percent
        stock_num = state_vec.shape[0]
        if Prev is None:
            init_F = np.expand_dims(softmax([random.random() for _ in range(stock_num)]), axis=0)
        else:
            inif_F = Prev["Fp"]
            feed_dict[self.Ap] = Prev["Ap"]
            feed_dict[self.Bp] = Prev["Bp"]
        feed_dcit[self.Fp] = init_F
        fetches = [self.A, self.B, self.F, self.grad_F_net_prev]
        A, B, F, grad_prev = session.run(fetches, feed_dict)
        return A, B, F, grad_prev

    def run_step(self, session, state_vec, rise_percent, Ap, Bp, Fp, weight, grad_prev, op):
        ''' 
        Stepwise Trading
        :param session: tensorflow session
        :param state_vec: state vector on stocks in trading days
        :param rise_percent: rise_percent of all stocks in trading days
        :param op: optimization option
        :return: portfolios, sharpe ratio, variable of policy net
        '''
        feed_dict = {}
        feed_dict[self.Ap] = float(Ap)
        feed_dict[self.Bp] = float(Bp)
        feed_dict[self.Fp] = Fp
        feed_dict[self.state_vec] = state_vec
        feed_dict[self.rise_percent] = rise_percent
        for i in range(len(grad_prev)):
            feed_dict[self.grad_prev[i]] = grad_prev[i]

        # For saving the gradient, but do no optimization
        fetches = [self.A, self.B, self.sharpe, self.F, self.grad_F_net_prev]
        A, B, sharpe, portfolio, grad_prev = session.run(fetches, feed_dict)
        # Do optimization
        fetches = [self.A, self.B, self.sharpe, self.F, op]
        A, B, sharpe, portfolio, _ = session.run(fetches, feed_dict)
        return A, B, sharpe, portfolio, grad_prev
    
    def run_epoch(self, session, state_vec, rise_percent, op, rd=True, Prev=None):
        '''
        Epoch-wise Runing
        :param session: tensorflow session
        :param state_vec: state vec of all stocks in trading days
        :param rise_percent: rise_percent of all stocks in trading days
        :param op: optimization option
        :return: Sharpe Ratio, Portfolios
        '''
        times, stock_num, fea_dim = features.shape
        portfolios = []
        sharpes = []
        total_reward = 0
        for time in range(times):
            state_vec_t = state_vec[time]
            rise_percent_t = np.expand_dims(rise_percent[time, :], axis=1)
            if time == 0:
                if Prev is None:
                    A, B, portfolio, grad_prev = self.run_init(session, state_vec_t, rise_percent_t)
                else:
                    A, B, portfolio, grad_prev = self.run_init(session, state_vec_t, rise_percent_t, Prev)
                sharpe = 0
            else:
                A, B, sharpe, portfolio, grad_prev = \
                        self.run_step(session, state_vec_t, rise_percent_t, A, B, portfolio, grad_prev, op)
            portfolios.append(portfolio)
            sharpes.append(sharpe)
            total_reward += np.dot(portfolio, rise_percent_t)
        return sharpes, portfolios, total_reward

    def run_test_epoch(self, session, state_vec, rise_percent):
        '''
        For test
        '''
        times, stock_num, fea_dim = state_vec.shape
        portfolios = []
        sharpes = []
        total_reward = 0
        for time in range(times):
            state_vec_t = state_vec[time]
            rise_percent_t = np.expand_dims(rise_percent[time, :], axis=1)
            feed_dict = {}
            feed_dict[self.state_vec] = state_vec_t
            feed_dict[self.rise_percent] = rise_percent_t
            if time == 0:
                A, B, portfolio, _ = self.run_init(session, state_vec_t, rise_percent_t)
            else:
                feed_dict[self.Ap] = float(A)
                feed_dict[self.Bp] = float(B)
                feed_dict[self.Fp] = portfolio
                fetches = [self.A, self.B, self.sharpe, self.F]
                A, B, sharpe, portfolio = session.run(fetches, feed_dict)

            sharpes.append(sharpe)
            portfolios.append(portfolio[0])
            total_reward += np.dot(portfolio, rise_percent_t)
        return sharpes, portfolios, total_reward
