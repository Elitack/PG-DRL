import tensorflow as tf
import numpy as np
import json
import argparse
from agent.RLagent import RLAgent
from agent.BaselineAgent import BaselineAgent

def parse_args():
    '''
    Parses the arguments.
    '''
    parser = argparse.ArgumentParser(description="input related param.")
    parser.add_argument('-set', nargs='?', help='Stock set')
    parser.add_argument('-out', nargs='?', help='Output file')
    parser.add_argument('-mode', nargs='?', help='Train or test')

    return parser.parse_args()

def main(args): 
    with open('data/stock_set.json', 'r') as f:
        stock_set = json.load(f)    
    if args.mode == 'train':
        config = {}
        config['cost'] = 0.003
        config['start_date'] = '20161226'
        config['split_date'] = '20170626'
        config['end_date'] = '20170926'
        config['lr'] = 0.001
        config['stocks'] = stock_set[args.set]
        config['stock_num'] = len(config['stocks'])

        for fea_dim in [3, 5, 10, 20]:
            for batch_feature in [5, 10, 20]:
                for batch_f in [4, 6, 11]:
                    for time_span in [240]:
                        tf.reset_default_graph()
                        config['fea_dim'] = fea_dim
                        config['batch_feature'] = batch_feature
                        config['batch_f'] = batch_f
                        config['time_span'] = time_span

                        try:
                            agent = RLAgent(config)

                            if not agent.valid:
                                continue
                            arg, result = agent.RL_train()
                            result = ' '.join(map(str, result))
                            output_file = '{}'.format(args.out)
                        except:
                            continue
                        with open(output_file, 'a+') as outf:
                            line = 'epoch:{}, fea_dim:{}, batch_feature:{}, batch_f:{}, time_span:{}, result:{} \n'.format(\
                                arg, fea_dim, batch_feature, batch_f, time_span, result)
                            outf.write(line)
                        print(line)


if __name__ == "__main__":
    args = parse_args()
    main(args)
   
    