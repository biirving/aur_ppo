import torch
from torch import nn, tensor
from nets import discrete_net, continuous_net, critic
from torch.distributions import MultivariateNormal, Categorical
import gym
import time
from actor_critic import actor_critic
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys, os, time

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

class torch_buffer():
	def __init__(self, observation_shape, action_shape, num_steps, num_envs):
		self.states = torch.zeros((num_steps, num_envs) +  observation_shape).to(device)
		self.actions = torch.zeros((num_steps, num_envs) + action_shape).to(device)
		self.log_probs = torch.zeros((num_steps, num_envs)).to(device)
		self.rewards = torch.zeros((num_steps, num_envs)).to(device)
		self.terminals = torch.zeros((num_steps, num_envs)).to(device)
		self.values = torch.zeros((num_steps, num_envs)).to(device)

# generalize training loop
# Debug this mf
class ppo():
	# add the metrics
	def __init__(self, params):
		self.batch_size = None
		self.minibatch_size = None
		# Define a list of attributes to exclude from direct assignment
		exclude_list = ['batch_size', 'minibatch_size']
		# Iterate through the keys in params
		for key, value in params.items():
			# Check if the key is not in the exclude list
			if key not in exclude_list:
				# Dynamically set the attribute based on the key-value pair
				setattr(self, key, value)
		self.batch_size = self.num_steps * self.num_envs
		self.minibatch_size = int(self.batch_size // self.num_minibatches)
		self.num_updates = self.total_timesteps // self.batch_size
		run_name = f"{self.gym_id}__{self.exp_name}__{self.seed}__{int(time.time())}"
		self.envs = gym.vector.SyncVectorEnv(
	    	[self.make_env(self.gym_id, int(self.seed) + i, i, params['capture_video'], run_name) for i in range(self.num_envs)]
	    )
		self.state_dim = self.envs.single_observation_space.shape
		if(self.continuous):
			self.action_dim = self.envs.single_action_space.shape
		else:
			self.action_dim = self.envs.single_action_space.n
		#print(self.env.single_action_space.shape)	
		self.policy = actor_critic(self.state_dim[0], 
			self.action_dim, self.hidden_dim, self.num_layers, self.dropout, self.continuous)

		# just keep them separate here
		self.policy_old = actor_critic(self.state_dim[0], 
			self.action_dim, self.hidden_dim, self.num_layers, self.dropout, self.continuous)
		self.policy_old.load_state_dict(self.policy.state_dict())

		self.buffer = torch_buffer(self.state_dim, self.envs.single_action_space.shape, self.num_steps, self.num_envs)
		self.optimizer =  torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate, eps=1e-5)
		self.total_returns = []
		self.total_episode_lengths = []
		self.x_indices = []

	def make_env(self, gym_id, seed, idx, capture_video, run_name):
		def thunk():
			env = gym.make(gym_id)
			env = gym.wrappers.RecordEpisodeStatistics(env)
			if capture_video:
				if idx == 0:
					env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
			if(self.continuous):
				env = gym.wrappers.ClipAction(env)
				env = gym.wrappers.NormalizeObservation(env)
				env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
				env = gym.wrappers.NormalizeReward(env)
				env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
			env.seed(seed)
			env.action_space.seed(seed)
			env.observation_space.seed(seed)
			return env
		return thunk

    # do the O(1) accesses slow down the code to a significant degree
	def rewards_to_go(self, step, next_obs, global_step):
		with torch.no_grad():
			#action, logprob, _, value= self.agent.get_action_and_value(next_obs)
			action, logprob, value = self.policy_old.act(next_obs)
			self.buffer.values[step] = value.flatten()
			self.buffer.actions[step] = action
			self.buffer.log_probs[step] = logprob
			next_obs, reward, done, info = self.envs.step(action.cpu().numpy())
			for item in info:
				if "episode" in item.keys():
					self.total_returns.append(item["episode"]["r"])
					self.total_episode_lengths.append(item["episode"]["l"])
					self.x_indices.append(global_step)
			self.buffer.rewards[step] = torch.tensor(reward).to(device).view(-1)
			# torch.Tensor outputs a Float tensor, while tensor.tensor infers a dtype
			next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
		return next_obs, next_done

	def advantages(self, next_obs, next_done):
		with torch.no_grad():
			next_value = self.policy_old.critic(next_obs).reshape(1, -1)
            # generalized advantage estimation
			if self.gae:
				advantages = torch.zeros_like(self.buffer.rewards).to(device)
				lastgaelam = 0
				for t in reversed(range(self.num_steps)):
					if t == self.num_steps - 1:
						nextnonterminal = 1.0 - next_done
						nextvalues = next_value
					else:
						nextnonterminal = 1.0 - self.buffer.terminals[t + 1]
						nextvalues = self.buffer.values[t + 1]
					# nextnonterminal is a mask
					delta = self.buffer.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.buffer.values[t]
					advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
				returns = advantages + self.buffer.values
			else:
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

	def train(self):
		#print(self.agent.actor[0].weight)
		#print(self.agent_new.actor[0].weight)
		#print('old policy', self.policy_old.actor.net[0].weight)
		#print('new policy', self.policy.actor.net[0].weight)
		global_step = 0
		start_time = time.time()
		next_obs = torch.Tensor(self.envs.reset()).to(device)
		next_done = torch.zeros(self.num_envs).to(device)
		policy_losses = []
		for update in range(1, self.num_updates + 1):
			if self.anneal_lr:
				frac = 1.0 - (update - 1.0) / self.num_updates
				lrnow = frac * self.learning_rate
				self.optimizer.param_groups[0]["lr"] = lrnow

			# check for bugs
			for step in range(0, self.num_steps):
				global_step += 1 * self.num_envs
				self.buffer.states[step] = next_obs
				self.buffer.terminals[step] = next_done
				next_obs, next_done = self.rewards_to_go(step, next_obs, global_step)		

			returns, advantages = self.advantages(next_obs, next_done)
			# we flatten here to have access to all of the environment returns
			b_obs = self.buffer.states.reshape((-1,) + self.envs.single_observation_space.shape)
			b_logprobs = self.buffer.log_probs.reshape(-1)
			b_actions = self.buffer.actions.reshape((-1,) + self.envs.single_action_space.shape)
			b_advantages = advantages.reshape(-1)
			b_returns = returns.reshape(-1)
			b_values = self.buffer.values.reshape(-1)
			b_inds = np.arange(self.batch_size)
			clip_fracs = []
			for ep in range(self.num_update_epochs):
				np.random.shuffle(b_inds)
				for index in range(0, self.batch_size, self.minibatch_size):
					mb_inds = b_inds[index:index+self.minibatch_size]
					_, newlogprob, entropy, newvalue = self.policy.evaluate(b_obs[mb_inds], b_actions.long()[mb_inds])
					log_ratio = newlogprob - b_logprobs[mb_inds]
					ratio = log_ratio.exp()
					# to check for early stopping. should remain below 0.2
					with torch.no_grad():
						old_approx_kl = (-log_ratio).mean()
						approx_kl = ((ratio - 1) - log_ratio).mean()
						clip_fracs += [((ratio - 1.0).abs() > self.clip_coeff).float().mean().item()]
					# we normalize at the minibatch level
					mb_advantages = b_advantages[mb_inds]
					# 1e-8 avoids division by 0
					if self.norm_adv:
						mb_advantages = (mb_advantages - mb_advantages.mean())/(mb_advantages.std() + 1e-8)

					# gradient descent, rather than ascent
					loss_one = -mb_advantages * ratio
					loss_two = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coeff, 1 + self.clip_coeff)
					policy_loss = torch.max(loss_one, loss_two).mean()
					# mean squared error
					#value_loss = 0.5 * (((newvalue.view(-1) - b_values[mb_inds])) ** 2).mean()
					#print(value_loss)
					print(newvalue.view(-1).shape)
					print(b_values[mb_inds].shape)
					value_loss = nn.MSELoss(newvalue.view(-1), b_values[mb_inds])

					# no value clipping here
					entropy_loss = entropy.mean()
 
					loss = policy_loss - self.entropy_coeff * entropy_loss + value_loss * self.value_coeff
					self.optimizer.zero_grad()

					loss.backward()
					nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
					self.optimizer.step()

				if self.target_kl is not None:
					if approx_kl > self.target_kl:
						break

			self.policy_old.load_state_dict(self.policy.state_dict())
			policy_losses.append(policy_loss.item())

		#torch.save(self.policy, 'actor_critic.pt')
		self.plot_episodic_returns(np.array(self.total_returns), np.array(np.array(self.x_indices)))
		#self.plot_episodic_returns(np.array(self.total_episode_lengths))

	def plot(self, loss, x_indices):
		print(loss.shape)
		#timesteps = np.arange(1, loss.shape[0] + 1)
		plt.plot(np.array(x_indices), loss)
		plt.xlabel('Timestep')
		plt.ylabel('Total returns')
		plt.title('Episode Length over time')
		plt.show()

	def moving_average(self, data, window_size):
		return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

	def plot_episodic_returns(self, episodic_returns, x_indices, window_size=10):
		print(len(episodic_returns))
		print(len(x_indices))
		smoothed_returns = self.moving_average(episodic_returns, window_size)
		plt.plot(x_indices, episodic_returns, label='Episodic Returns')
		plt.plot(x_indices[9:], smoothed_returns, label=f'Moving Average (Window Size = {window_size})', color='red')
		plt.title('Episodic Returns with Moving Average for Cartpole Problem')
		plt.xlabel('Timestep')
		plt.ylabel('Return')
		plt.legend()
		plt.show()
