from torch import nn, tensor
#import gymnasium as gym
import gym
import torch
from torch import nn, tensor
from torch.distributions import Categorical
from tqdm import tqdm
import matplotlib.pylab as plt
import numpy as np
from collections import deque
from ppo import ppo


device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

# a class to run our trained models
class test():
    def __init__(self, model, env, render_mode):
        self.env = gym.make(env)
        input_dim = self.env.observation_space.shape[0]
        output_dim = self.env.action_space.n
        self.agent = torch.load(model)

    def moving_average(self, data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    def plot_episodic_returns(self, episodic_returns, x_indices, window_size=10):
        smoothed_returns = self.moving_average(episodic_returns, window_size)
        plt.plot(x_indices, episodic_returns, label='Episodic Returns')
        plt.plot(x_indices[9:], smoothed_returns, label=f'Moving Average (Window Size = {window_size})', color='red')
        plt.title('Episode lengths')
        plt.xlabel('Episode')
        plt.ylabel('Length')
        plt.legend()
        plt.show()

    def run(self, episodes, max_length=10000):
        #optim = torch.optim.Adam(self.net.parameters(), lr=5e-5)
        all_rewards = []
        all_loss = []
        episode_lenghts = []
        for _ in tqdm(range(episodes)):
            self.agent.zero_grad()
            state = self.env.reset()
            state = torch.from_numpy(state)
            log_probs = []
            rewards = []
            # run the episode
            with torch.no_grad():
                for _ in range(max_length):
                    action, log_prob, value  = self.agent.act(state.to(device)) 
                    next_state, reward, is_terminal, info = self.env.step(action.item())
                    rewards.append(reward)
                    if is_terminal:
                        break
                state = torch.from_numpy(next_state)
            episode_lenghts.append(len(rewards))
        # episode lengths
        self.plot_episodic_returns(np.array(episode_lenghts), np.arange(len(episode_lenghts)))
test = test('actor_critic.pt', 'CartPole-v1', render_mode=None)
test.run(1000)



