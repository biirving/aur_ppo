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


class ppo():
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

    # do the O(1) accesses slow down the code to a significant degree
	#@profile
	def rewards_to_go(self, step, next_obs, global_step, writer):
		with torch.no_grad():
			action, logprob, _, value = self.policy.evaluate(next_obs.to(device))
			self.buffer.values[step] = value.flatten()
		self.buffer.actions[step] = action
		self.buffer.log_probs[step] = logprob

		next_obs, reward, done, _, info = self.envs.step(action.cpu().numpy())
		self.buffer.rewards[step] = torch.tensor(reward).view(-1)
		next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

		if('final_info' in info.keys()):
			for item in info['final_info']:
				if item is not None:
					writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
					writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
					self.total_returns.append(item["episode"]["r"])
					self.total_episode_lengths.append(item["episode"]["l"])
					self.x_indices.append(global_step)
					break
		return next_obs, next_done

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

	def advantages(self, next_obs, next_done):
		with torch.no_grad():
			next_value = self.policy.value(next_obs)
			if self.gae:
				returns, advantages = self.run_gae(next_value, next_done)
			else:
				returns, advantages = self.normal_advantage(next_value, next_done)
		return returns, advantages

	#@profile
	def train(self):
		if self.track:
			import wandb
			wandb.init(project='ppo',entity='Aurelian',sync_tensorboard=True,config=None,name=self.run_name,monitor_gym=True,save_code=True)
		writer = SummaryWriter(f"runs/{self.run_name}")
		writer.add_text("parameters/what", "what")
		writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{str(self.params_dict[key])}|" for key in self.params_dict])),
    	)

		seed = 1
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.backends.cudnn.deterministic = True 

		global_step = 0
		start_time = time.time()
		next_obs = torch.Tensor(self.envs.reset(seed=list(range(self.num_envs)))[0]).to(device)
		next_done = torch.zeros(self.num_envs).to(device)
		policy_losses = []
		#sys.exit()
		for update in tqdm(range(1, self.num_updates + 1)):
			t0 = time.time()
			# adjust learning rate
			if self.anneal_lr:
				frac = 1.0 - (update - 1.0) / self.num_updates
				lrnow = frac * self.learning_rate
				self.optimizer.param_groups[0]["lr"] = lrnow

			# generate rewards to go for environment before update 
			for step in range(0, self.num_steps):
				global_step += 1 * self.num_envs
				self.buffer.states[step] = next_obs
				self.buffer.terminals[step] = next_done
				next_obs, next_done = self.rewards_to_go(step, next_obs, global_step, writer)	
								
			returns, advantages = self.advantages(next_obs, next_done)
			print(advantages.shape)

			(b_obs, b_logprobs, b_actions, 
				b_advantages, b_returns, b_values) = self.buffer.flatten(returns, advantages)

			b_inds = np.arange(self.batch_size)
			clip_fracs = []
			for ep in range(self.num_update_epochs):
				# we randomize before processing the minibatches
				np.random.shuffle(b_inds)
				for index in range(0, self.batch_size, self.minibatch_size):
					mb_inds = b_inds[index:index+self.minibatch_size]
					_, newlogprob, entropy, newvalue = self.policy.evaluate(b_obs[mb_inds], b_actions[mb_inds])

					# clamp the new and old log probabilities
					#newlogprob = torch.clamp(newlogprob, 1e-10, 1.0)
					#old_log_probs = torch.clamp(b_logprobs[mb_inds], 1e-10, 1.0)
					old_log_probs = b_logprobs[mb_inds]
					log_ratio = newlogprob - old_log_probs 

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
					print('ratio', ratio.shape)
					print('advantages', mb_advantages.shape)
					loss_one = -mb_advantages * ratio
					loss_two = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coeff, 1 + self.clip_coeff)
					policy_loss = torch.max(loss_one, loss_two).mean()
					policy_losses.append(policy_loss.item())

					# value clipping
					newvalue = newvalue.view(-1)
					if self.clip_vloss:
						v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
						v_clipped = b_values[mb_inds] + torch.clamp(
							newvalue - b_values[mb_inds],
							-self.clip_coeff,
							self.clip_coeff,
						)
						v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
						v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
						value_loss = 0.5 * v_loss_max.mean()
					else:
						value_loss = 0.5 * ((newvalue - b_values[mb_inds]) ** 2).mean()

					entropy_loss = entropy.mean()
					loss = policy_loss - self.entropy_coeff * entropy_loss + value_loss * self.value_coeff

					self.optimizer.zero_grad()
					loss.backward()
					nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
					self.optimizer.step()

				if self.target_kl is not None:
					if approx_kl > self.target_kl:
						break

			policy_losses.append(policy_loss.item())

			y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
			var_y = np.var(y_true)
			explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

			# should be done in a separate function
			# Writer object?
			writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
			writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
			writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
			writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
			writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
			writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
			writer.add_scalar("losses/clipfrac", np.mean(clip_fracs), global_step)
			writer.add_scalar("losses/explained_variance", explained_var, global_step)
			#print("SPS:", int(global_step / (time.time() - start_time)))
			writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

		self.envs.close()
		writer.close()
		torch.save(self.policy, 'actor_critic_' + str(self.num_layers) + '.pt')
		self.plot_episodic_returns(np.array(self.total_returns), np.array(np.array(self.x_indices)), 'episodic returns')
		self.plot_episodic_returns(np.array(self.total_episode_lengths), np.array(np.array(self.x_indices)), 'episodic lengths')

		return self.total_returns, self.total_episode_lengths, self.x_indices
		#self.plot_episodic_returns(np.array(policy_losses), np.arange(len(policy_losses)))

	def plot(self, loss, x_indices):
		plt.plot(np.array(x_indices), loss)
		plt.xlabel('Timestep')
		plt.ylabel('Total returns')
		plt.title('Episode Length over time')
		plt.show()

	def moving_average(self, data, window_size):
		return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

	def plot_episodic_returns(self, episodic_returns, x_indices, title, window_size=10):
		smoothed_returns = self.moving_average(episodic_returns, window_size)
		plt.plot(x_indices, episodic_returns, label='Episodic Returns')
		plt.plot(x_indices[9:], smoothed_returns, label=f'Moving Average (Window Size = {window_size})', color='red')
		plt.title('Episodic Returns with Moving Average for ' + self.gym_id)
		plt.xlabel('Timestep')
		plt.ylabel('Return')
		plt.legend()
		plt.savefig('../plots/' + title + '_num_layers_' + str(self.num_layers) + '_dropout_' + str(self.dropout) + '_num_envs_' + str(self.num_envs) + '_num_mb_' + str(self.num_minibatches) + '.png')
