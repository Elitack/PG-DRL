import tensorflow as tf
import numpy as np
import json
import argparse
from RLAgent import RLAgent
import sys

'''
def parse_args():

    parser = argparse.ArgumentParser(description="input related param.")
    # parser.add_argument('-set', nargs='?', help='Stock set')
    # parser.add_argument('-out', nargs='?', help='Output file')
    # parser.add_argument('-mode', nargs='?', help='Train or test')

    return parser.parse_args()
'''


def main(arg):
    config = dict()
    config['lr'] = 0.0000001
    config['stocks'] = ['a', 'aa']
    config['stock_num'] = len(config['stocks'])
    config['date'] = arg

    tf.reset_default_graph()

    agent = RLAgent(config)
    agent.RL_train()


if __name__ == "__main__":
    # args = parse_args()
    main(sys.argv[1])
