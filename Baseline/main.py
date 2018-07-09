import tensorflow as tf
import numpy as np
import json
import argparse
from BaselineAgent import BaselineAgent


def main():
    config = dict()
    config['lr'] = 0.0000001
    config['stocks'] = ['a', 'aa']
    config['stock_num'] = len(config['stocks'])

    tf.reset_default_graph()

    agent = RLAgent(config)
    agent.RL_train()


if __name__ == "__main__":
    # args = parse_args()
    main()
