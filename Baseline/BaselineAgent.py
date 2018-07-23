import tensorflow as tf
import numpy as np
from tool import *
from Data import DM
from cvxopt import solvers, matrix

class BaselineAgent(object):
    def __init__(self, config):
        self.DM = DM()
        self.config = config
        self.DM.input_range(DATE[1][0], DATE[1][1], self.config['stocks'])
        self.train_fea, self.train_rp, self.train_p = self.DM.gen_data()
        self.DM.input_range(DATE[1][1], DATE[1][2], self.config['stocks'])
        self.test_fea, self.test_rp, self.test_p = self.DM.gen_data()        

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
        return    

    def metrics(self, seq, days):
        length = len(seq)
        ret = seq[1:] - seq[:-1]
        ret_rate = (seq[1:] - seq[:-1]) / seq[:-1]

        ar = seq[-1] / seq[0]
        ar = np.power(ar, 240 / days) - 1
        sr = ret_rate.mean() / ret_rate.std()
        vol = ret.std()
        ir = ret.mean() / ret.std()
        md = -ret_rate.min()

        return [round(ar, 4), round(sr, 4), round(vol, 4), round(ir, 4), round(md, 4)]

    def best_stock(self):
        portfolios = np.zeros(self.test_rp.shape)
        time_step = portfolios.shape[0]
        stock_num = portfolios.shape[1]

        for t in range(time_step):
            fea = self.test_fea[t]
            pre_rp = (fea[:, -1] - fea[:, 0]) / fea[:, 0]
            select_stock = np.argmax(pre_rp)
            portfolios[t][select_stock] = 1
        self.evaluation(self.test_p, portfolios, time_step)
        
        return 

    def UBAH(self):
        portfolios = np.zeros(self.test_rp.shape)
        time_step = portfolios.shape[0]
        stock_num = portfolios.shape[1]      

        portfolios[0] = np.ones(stock_num) / stock_num
        for t in range(1, time_step):
            new_f = portfolios[t-1]*(self.test_rp[t-1]+1)
            new_f = new_f / np.sum(new_f)
            portfolios[t] = new_f
        self.evaluation(self.test_p, portfolios, time_step)
        
        return        

    def UCRP(self):
        portfolios = np.zeros(self.test_rp.shape)
        time_step = portfolios.shape[0]
        stock_num = portfolios.shape[1]

        portfolios = np.ones(portfolios.shape) / stock_num

        self.evaluation(self.test_p, portfolios, time_step)
        
        return  

    def eg(self):
        portfolios = np.zeros(self.test_rp.shape)
        time_step = portfolios.shape[0]
        stock_num = portfolios.shape[1]

        portfolios[0] = np.ones(self.test_rp.shape[1]) / self.test_rp.shape[1]

        for i in range(1, time_step):
            b = portfolios[i-1] * np.exp(0.05 * self.test_rp[i-1] / np.dot(self.test_rp[i-1], portfolios[i-1]))
            b = b / np.sum(b)
            portfolios[i] = b
        self.evaluation(self.test_p, portfolios, time_step)


    def olmar(self):
        portfolios = np.zeros(self.test_rp.shape)
        time_step = portfolios.shape[0]
        stock_num = portfolios.shape[1]
   
        last_b = np.ones(self.test_rp.shape[1]) / self.test_rp.shape[1]   
        
        for i in range(time_step): 
            if i == 0:
                data_phi = np.ones((1, self.test_rp.shape[1])) 
            else:
                data_phi = 0.5 + (1 - 0.5) * data_phi / (1 + self.test_rp[i-1])
            ell = max(0, 10 - data_phi.dot(last_b))
            x_bar = data_phi.mean()
            denominator = np.linalg.norm(data_phi - x_bar)**2
            if denominator == 0:
                lam = 0
            else:
                lam = ell / denominator
            data_phi = np.squeeze(data_phi)
            b = last_b + lam * (data_phi - x_bar)
            b = self.euclidean_proj_simplex(b)
            portfolios[i] = b
            last_b = b
        self.evaluation(self.test_p, portfolios, time_step)

    def ons(self):
        portfolios = np.zeros(self.test_rp.shape)
        time_step = portfolios.shape[0]
        stock_num = portfolios.shape[1]

        last_b = np.ones(self.test_rp.shape[1]) / self.test_rp.shape[1]   
           
        m = self.test_rp.shape[1]
        A = np.mat(np.eye(m))
        b = np.mat(np.zeros(m)).T

        for i in range(time_step):       
            if i == 0:
                x = np.ones(self.test_rp.shape[1])
            else:
                x = 1 + self.test_rp[i-1]
            grad = np.mat(x / np.dot(last_b, x)).T
            A += grad * grad.T
            b += 2 * grad
            pp = self.projection_in_norm(0.125 * A.I * b, A)
            portfolios[i] = pp
            last_b = pp
        self.evaluation(self.test_p, portfolios, time_step)

    def cornk(self):
        gamma=0.25
        
        portfolios = np.zeros(self.test_rp.shape)
        time_step = portfolios.shape[0]
        stock_num = portfolios.shape[1] 
        
        last_b = np.ones(self.test_rp.shape[1]) / self.test_rp.shape[1]

        for i in range(time_step):
            b = last_b * (1 - gamma - gamma / self.test_rp.shape[1]) + gamma / self.test_rp.shape[1]
            b = b / np.sum(b)
            last_b = b
            portfolios[i] = b
        self.evaluation(self.test_p, portfolios, time_step)

    def euclidean_proj_simplex(self, v, s=1):
        '''Compute the Euclidean projection on a positive simplex
        :param v: n-dimensional vector to project
        :param s: int, radius of the simple
        return w numpy array, Euclidean projection of v on the simplex
        Original author: John Duchi
        '''
        assert s>0, "Radius s must be positive (%d <= 0)" % s

        n, = v.shape # raise ValueError if v is not 1D
        # check if already on the simplex
        if v.sum() == s and np.alltrue( v>= 0):
            return v

        # get the array of cumulaive sums of a sorted copy of v
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        # get the number of >0 components of the optimal solution
        rho = np.nonzero(u * np.arange(1, n+1) >= (cssv - s))[0][-1]
        # compute the Lagrange multiplier associated to the simplex constraint
        theta = (cssv[rho] - s) / (rho + 1.)
        w = (v-theta).clip(min=0)
        return w

    def projection_in_norm(self, x, M):
        """ Projection of x to simplex indiced by matrix M. Uses quadratic programming.
        """
        m = M.shape[0]

        P = matrix(2*M)
        q = matrix(-2 * M * x)
        G = matrix(-np.eye(m))
        h = matrix(np.zeros((m,1)))
        A = matrix(np.ones((1,m)))
        b = matrix(1.)

        sol = solvers.qp(P, q, G, h, A, b)
        return np.squeeze(sol['x'])