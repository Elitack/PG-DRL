import argparse
import numpy as np
from agent import Agent
from tool import *
from ddpg_model import Model
from env import Env

config = {}

def parse_args():
    '''
    Parses the arguments.
    '''
    parser = argparse.ArgumentParser(description="input related param.")
    parser.add_argument('-set', nargs='?', help='Stock set')
    parser.add_argument('-out', nargs='?', help='Output file')
    parser.add_argument('-mode', nargs='?', help='Train or test')

def main():
    env = Env(config)
    state_dim = (batch_prev+1, env.fea.shape[1], env.fea.shape[2], 3)
    action_dim = env.stock_num
    model = Model(state_dim,
                  action_dim,
                  actor_learning_rate=ACTOR_LEARNING_RATE,
                  critic_learning_rate=CRITIC_LEARNING_RATE,
                  tau=TAU)
    replay_buffer = Replay_Buffer(buffer_size=int(BUFFER_SIZE), batch_size=BATCH_SIZE)
    exploration_noise = OU_Process(action_dim)
    agent = Agent(model, replay_buffer, exploration_noise, discout_factor=DISCOUNT_FACTOR)

    for episode in range(EPISODES):
        env.reset()
        # Training:
        for ti in range(EXPLORE_TIMES):
            agent.init_process()
            state, prev_f = env.select_batch()
            for step in range(batch_f):
                # env.render()
                if episode < MAX_EXPLORE_EPS:
                    p = episode / MAX_EXPLORE_EPS
                    action = agent.select_action(prev_f, state, p)
                else:
                    action = agent.predict_action(prev_f, state)
                next_state, reward = env.step(action)
                agent.store_transition([prev_f, state, action, reward, next_state])
                if agent.replay_buffer.memory_state()["current_size"] > WARM_UP_MEN:
                    agent.train_model()

        # Testing:
        '''
        if episode >= MAX_EXPLORE_EPS:
            total_reward = 0
            for i in range(TEST_EPS):
                state = env.reset()
                for j in range(env.spec.timestep_limit):
                    # env.render()
                    state = np.reshape(state, (1, 3))
                    action = agent.predict_action(state)
                    action_ = action * 2
                    state, reward, done, _ = env.step(action_)
                    total_reward += reward
                    if done:
                        break
            avg_reward = total_reward/TEST_EPS
            print("episode: {}, Evaluation Average Reward: {}".format(episode, avg_reward))
        '''


if __name__ == '__main__':
    main()
