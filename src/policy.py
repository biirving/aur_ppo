import torch
from torch import nn, tensor
from nets import discrete_net, continuous_net, critic
from torch.distributions import MultivariateNormal, Categorical
import gym
import time
from models import actor_critic
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys, os, time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
from abc import ABC, abstractmethod 

from torch.distributions import Normal, Categorical

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class torch_buffer():
	def __init__(self, observation_shape, action_shape, num_steps, num_envs):
		self.observation_shape = observation_shape
		self.action_shape = action_shape
		self.states = torch.zeros((num_steps, num_envs) + observation_shape).to(device)
		self.actions = torch.zeros((num_steps, num_envs) + action_shape).to(device)
		self.log_probs = torch.zeros((num_steps, num_envs)).to(device)
		self.rewards = torch.zeros((num_steps, num_envs)).to(device)
		self.terminals = torch.zeros((num_steps, num_envs)).to(device)
		self.values = torch.zeros((num_steps, num_envs)).to(device)

	# flatten the buffer values for evaluation
	def flatten(self, returns, advantages):
		b_obs = self.states.reshape((-1,) + self.observation_shape)
		b_logprobs = self.log_probs.reshape(-1)
		b_actions = self.actions.reshape((-1,) + self.action_shape)
		b_advantages = advantages.reshape(-1)
		b_returns = returns.reshape(-1)
		b_values = self.values.reshape(-1)
		return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values


class policy(ABC):
	# add the metrics
	def __init__(self, params):
		# accessing through the dictionary is slower
		self.params_dict = params

		self.all_steps = None
		self.minibatch_size = None
		# Define a list of attributes to exclude from direct assignment
		exclude_list = ['batch_size', 'minibatch_size']
		# Iterate through the keys in params
		for key, value in params.items():
			# Check if the key is not in the exclude list
			if key not in exclude_list:
				# Dynamically set the attribute based on the key-value pair
				setattr(self, key, value)

		# Tht total steps x the number of envs represents how many total
		# steps in the said environment will be taken by the training loop		
		self.all_steps = self.num_steps * self.num_envs
		self.batch_size = int(self.num_envs * self.num_steps)
		self.minibatch_size = int(self.all_steps // self.num_minibatches)
		self.num_updates = self.total_timesteps // self.batch_size
		self.run_name = f"{self.gym_id}__{self.exp_name}__{self.seed}__{int(time.time())}"
		self.envs = gym.vector.SyncVectorEnv(
	    	[self.make_env(self.gym_id, i, params['capture_video']) for i in range(self.num_envs)]
	    )
		#self.envs = gym.vector.make(self.gym_id, num_envs=self.num_envs)
		self.state_dim = self.envs.single_observation_space.shape
		if(self.continuous):
			self.action_dim = self.envs.single_action_space.shape
		else:
			self.action_dim = self.envs.single_action_space.n

		self.policy = actor_critic(self.state_dim[0], 
			self.action_dim, self.hidden_dim, self.num_layers, self.dropout, self.continuous).to(device)

		self.buffer = torch_buffer(self.state_dim, self.envs.single_action_space.shape, self.num_steps, self.num_envs)
		self.optimizer =  torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate, eps=1e-5)
		self.total_returns = []
		self.total_episode_lengths = []
		self.x_indices = []

	def make_env(self, gym_id, idx, capture_video):
		def thunk():
			env = gym.make(gym_id)
			env = gym.wrappers.RecordEpisodeStatistics(env)
			if capture_video:
				if idx == 0:
					env = gym.wrappers.RecordVideo(env, f"videos/{self.run_name}")
			if(self.continuous):
				env = gym.wrappers.ClipAction(env)
				env = gym.wrappers.NormalizeObservation(env)
				env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
				env = gym.wrappers.NormalizeReward(env)
				env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
			return env
		return thunk


	def run_gae(self, next_value, next_done):
		advantages = torch.zeros_like(self.buffer.rewards)
		lastgaelam = 0
		for t in reversed(range(self.num_steps)):
			if t == self.num_steps - 1:
				nextnonterminal = 1.0 - next_done
				nextvalues = next_value
			else:
				nextnonterminal = 1.0 - self.buffer.terminals[t + 1]
				nextvalues = self.buffer.values[t + 1]
			# If episode finishes after timestep t, we mask the value and the previous advantage value, 
			# so that advantages[t] stores the (reward - value) at that timestep without
			# taking a value from the next episode, or storing a value in this episode to be multiplied
			# in the calculation of the rollout of the next episode
			delta = self.buffer.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.buffer.values[t]
			advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
		returns = advantages + self.buffer.values
		return returns, advantages

	# these can be stored in a separate file
	def normal_advantage(self, next_value, next_done):
		# the default rollout scheme
		returns = torch.zeros_like(self.buffer.rewards).to(device)
		for t in reversed(range(self.num_steps)):
			if t == self.num_steps - 1:
				nextnonterminal = 1.0 - next_done
				next_return = next_value
			else:
				nextnonterminal = 1.0 - self.buffer.terminals[t + 1]
				next_return = returns[t + 1]
			returns[t] = self.buffer.rewards[t] + self.gamma * nextnonterminal * next_return
		advantages = returns - self.buffer.values
		return returns, advantages

	@abstractmethod
	def rewards_to_go(self):
		pass

	def advantages(self, next_obs, next_done):
		with torch.no_grad():
			next_value = self.policy.value(next_obs)
			if self.gae:
				returns, advantages = self.run_gae(next_value, next_done)
			else:
				returns, advantages = self.normal_advantage(next_value, next_done)
		return returns, advantages

	@abstractmethod
	def train(self):
		pass
		
	def plot(self, loss, x_indices):
		#timesteps = np.arange(1, loss.shape[0] + 1)
		plt.plot(np.array(x_indices), loss)
		plt.xlabel('Timestep')
		plt.ylabel('Total returns')
		plt.title('Episode Length over time')
		plt.show()

	def moving_average(self, data, window_size):
		return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

	# add chart titles and all of that stuff
	def plot_episodic_returns(self, episodic_returns, x_indices, title, window_size=10):
		smoothed_returns = self.moving_average(episodic_returns, window_size)
		plt.plot(x_indices, episodic_returns, label='Episodic Returns')
		plt.plot(x_indices[9:], smoothed_returns, label=f'Moving Average (Window Size = {window_size})', color='red')
		plt.title('Episodic Returns with Moving Average for ' + self.gym_id)
		plt.xlabel('Timestep')
		plt.ylabel('Return')
		plt.legend()
		plt.savefig('/home/benjamin/Desktop/ml/aur_ppo/plots/' + title + '_num_layers_' + str(self.num_layers) + '_dropout_' + str(self.dropout) + '_num_envs_' + str(self.num_envs) + '_num_mb_' + str(self.num_minibatches) + '.png')