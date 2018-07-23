import random
import numpy as np
from collections import deque

ENV_NAME = 'Pendulum-v0'
EPISODES = 200

FEATURE_DIM = 10

batch_f = 10
batch_prev = 10
batch_feature = batch_f + batch_prev
cost = 0.003
gamma = 0.95


class OU_Process(object):
    def __init__(self, action_dim, theta=0.15, mu=0, sigma=0.2):
        self.action_dim = action_dim
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.current_x = None

        self.init_process()

    def init_process(self):
        self.current_x = np.ones(self.action_dim) * self.mu

    def update_process(self):
        dx = self.theta * (self.mu - self.current_x) + self.sigma * np.random.randn(self.action_dim)
        self.current_x = self.current_x + dx

    def return_noise(self):
        self.update_process()
        scale_x = (self.current_x - self.current_x.min()) / (self.current_x.max() - self.current_x.min())
        norm_x = scale_x / np.sum(scale_x)
        return norm_x


class Replay_Buffer(object):
    def __init__(self, buffer_size=10e6, batch_size=1):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)

    def __call__(self):
        return self.memory

    def store_transition(self, transition):
        self.memory.append(transition)

    def store_transitions(self, transitions):
        self.memory.extend(transitions)

    def get_batch(self, batch_size=None):
        b_s = batch_size or self.batch_size
        cur_men_size = len(self.memory)
        if cur_men_size < b_s:
            return random.sample(list(self.memory), cur_men_size)
        else:
            return random.sample(list(self.memory), b_s)

    def memory_state(self):
        return {"buffer_size": self.buffer_size,
                "current_size": len(self.memory),
                "full": len(self.memory)==self.buffer_size}

    def empty_transition(self):
        self.memory.clear()


def min_max_norm(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x