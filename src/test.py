from torch import nn, tensor
import gymnasium as gym
import torch
from torch import nn, tensor
from torch.distributions import Categorical
from tqdm import tqdm
import matplotlib.pylab as plt
import numpy as np
from collections import deque
from ppo import ppo


device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

class reinforce():
    def __init__(self, gamma:float, alpha:float, env, dim:int, layers:int, dropout:float):
        self.env = gym.make(env)
        input_dim = self.env.observation_space.shape[0]
        output_dim = self.env.action_space.n
        #self.net = Net(dim, input_dim, output_dim, layers, dropout).to(device)
        #self.net = torch.load('reinforce.pt')
        self.agent = torch.load('actor_critic.pt')
        self.alpha = alpha
        self.gamma = gamma
    def ema_plot(self, loss):
        timesteps = np.arange(1, loss.shape[0] + 1)
        alpha = 0.2  # Adjust this smoothing factor as needed
        ema = [loss[0]]
        for r in loss[1:]:
            ema.append(alpha * r + (1 - alpha) * ema[-1])
        plt.plot(timesteps, ema)
        plt.xlabel('Timestep')
        plt.ylabel('Neg log prob loss')
        plt.title('Loss')
        plt.show()
        #plt.close()
    def plot(self, loss):
        timesteps = np.arange(1, loss.shape[0] + 1)
        plt.plot(timesteps, loss)
        plt.xlabel('Timestep')
        plt.ylabel('Episode length')
        plt.title('Episode Length over time')
        plt.show()
    def run(self, episodes, max_length=10000):
        #optim = torch.optim.Adam(self.net.parameters(), lr=5e-5)
        all_rewards = []
        all_loss = []
        episode_lenghts = []
        for _ in tqdm(range(episodes)):
            self.agent.zero_grad()
            state, _ = self.env.reset()
            state = torch.from_numpy(state)
            log_probs = []
            rewards = []
            # run the episode
            for _ in range(max_length):
                action, log_prob, _ = self.agent.act(state.to(device)) 
                next_state, reward, is_terminal, _, _ = self.env.step(action.item())
                log_probs.append(log_prob)
                rewards.append(reward)
                if is_terminal:
                    break
                state = torch.from_numpy(next_state)
            episode_lenghts.append(len(rewards))
        self.plot(np.array(episode_lenghts)) 

test = reinforce(0.9, 1.0, "CartPole-v1", 64, 2, 0.0)
test.run(10000)
