import torch
from torch import nn, tensor
from nets import discrete_net, continuous_net, critic
from torch.distributions import MultivariateNormal, Categorical
import gym
import time
from models import robot_actor_critic
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys, os, time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
from env_wrapper import EnvWrapper
from torch.distributions import Normal, Categorical

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class torch_buffer():
	def __init__(self, state_shape, observation_shape, action_shape, num_steps, num_envs):
		self.state_shape = state_shape
		self.observation_shape = observation_shape
		self.action_shape = action_shape
		self.states = torch.zeros((num_steps, num_envs)).to(device)
		self.observations = torch.zeros((num_steps, num_envs) + observation_shape)
		self.actions = torch.zeros((num_steps, num_envs, action_shape)).to(device)
		self.log_probs = torch.zeros((num_steps, num_envs, action_shape)).to(device)
		self.rewards = torch.zeros((num_steps, num_envs)).to(device)
		self.terminals = torch.zeros((num_steps, num_envs)).to(device)
		self.values = torch.zeros((num_steps, num_envs)).to(device)

	# flatten the buffer values for evaluation
	def flatten(self, returns, advantages):
		b_states = self.states.view(self.states.shape[0] * self.states.shape[1])
		b_obs = self.observations.view(self.observations.shape[0] * self.observations.shape[1], 
								 self.observations.shape[2], self.observations.shape[3], self.observations.shape[4])
		b_logprobs = self.log_probs.reshape(-1)
		b_actions = self.actions.view(self.actions.shape[0] * self.actions.shape[1], self.actions.shape[2])
		b_advantages = advantages.reshape(-1)
		b_returns = returns.reshape(-1)
		b_values = self.values.reshape(-1)
		return b_states, b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values

# simple class for plotting in this environment
class store_returns():
	def __init__(self, num_envs, gamma):
		self.gamma = gamma
		self.env_returns = [[] for _ in range(num_envs)]
	
	def add_value(self, i, reward):
		self.env_returns[i].append(reward)

	def calc_discounted_return(self, i):
		len_episode = len(self.env_returns[i])
		R = 0
		for r in self.env_returns[i][::-1]:
			R = r + self.gamma * R
		self.env_returns[i] = []
		return R, len_episode



class robot_ppo():
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

		num_eval_processes=5
		simulator='pybullet'
		env=self.gym_id
		# config used in Dian, Rob, and Robin's paper
		env_config={'workspace': np.array([[ 0.25,  0.65],
			[-0.2 ,  0.2 ],
			[ 0.01,  0.25]]), 'max_steps': 100, 
			'obs_size': 128, 
			'fast_mode': True, 
			'action_sequence': 'pxyzr', 
			'render': False, 
			'num_objects': 2, 
			'random_orientation': True, 
			'robot': 'kuka', 
			'workspace_check': 'point', 
			'object_scale_range': (1, 1), 
			'hard_reset_freq': 1000, 
			'physics_mode': 'fast', 
			'view_type': 'camera_center_xyz', 
			'obs_type': 'pixel', 
			'view_scale': 1.5, 
			'transparent_bin': True}
		planner_config={'random_orientation': True, 'dpos': 0.02, 'drot': 0.19634954084936207}
		self.envs = EnvWrapper(self.num_envs, simulator, env, env_config, planner_config)
		self.eval_envs = EnvWrapper(num_eval_processes, simulator, env, env_config, planner_config)
		self.plot_index = 0 

		
		self.policy = robot_actor_critic(device, self.equivariant).to(device)

		# for close loop block pulling
		self.action_dim = 5
		self.state_dim = 1 
		# working with 128 x 128 images
		self.buffer = torch_buffer(self.state_dim, (1, 128, 128), self.action_dim, self.num_steps, self.num_envs)
		self.optimizer =  torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate, eps=1e-5)
		# We use some imitation learning on the actors parameters
		self.pretrain_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=self.learning_rate, eps=1e-5)
		self.total_returns = []
		self.total_episode_lengths = []
		self.x_indices = []
		self.episodic_returns = store_returns(self.num_envs, self.gamma)

	def rewards_to_go(self, step, next_state, next_obs, global_step, writer):
		with torch.no_grad():
			# will just be exploring randomly, without any pretraining/augmentation
			actions, unscaled, logprob, _, value = self.policy.evaluate(next_state.to(device), next_obs.to(device))
			self.buffer.values[step] = value.tensor.flatten()
		self.buffer.actions[step] = actions
		self.buffer.log_probs[step] = logprob

		next_states, next_obs, reward, done = self.envs.step(actions, auto_reset=True)

		for i, rew in enumerate(reward):
			self.episodic_returns.add_value(i, rew)

		self.buffer.rewards[step] = torch.tensor(reward).view(-1)
		next_states, next_obs, next_done = torch.tensor(next_states), torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

		# get reward of the local episode
		# we have to 'book keep' at each step
		for i, d in enumerate(done):
			if d:
				discounted_return, episode_length = self.episodic_returns.calc_discounted_return(i)
				# then we get the discounted episodic return for env 'i'
				writer.add_scalar("charts/discounted_episodic_return", discounted_return, self.plot_index)
				writer.add_scalar("charts/episodic_length", episode_length, self.plot_index)
				self.plot_index += 1
				break
				#how do we deal with multiple episodes ending on the same global step?

		return next_states, next_obs, next_done

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
			next_value = self.policy.value(next_obs).tensor.flatten()
			if self.gae:
				returns, advantages = self.run_gae(next_value, next_done)
			else:
				returns, advantages = self.normal_advantage(next_value, next_done)
		return returns, advantages

	def normalizeTransition(self, obs):
		#obs = torch.clip(obs, 0, 0.32)
		#obs = obs/0.4*255
		#obs = obs.to(torch.uint8)
		return obs

	#simple imitation learning pretraining of our agent
	# is this completely wrong
	def pretrain(self):
		loss_fct = torch.nn.MSELoss()
		state, obs = self.envs.reset()
		# we want to pretrain our policy... use simple MSE loss
		# using the distributed environment?
		index = 0
		p = 0
		while p < self.pretrain_episodes:
			obs = self.normalizeTransition(obs)
			# the ground truth action for our environments
			true_actions = self.envs.getNextAction()
			_, scaled_true_actions = self.policy.getActionFromPlan(true_actions)
			agent_action, _, _, _, _ = self.policy.evaluate(state.to(device), obs.to(device))
			state, obs, _, done = self.envs.step(scaled_true_actions, auto_reset=True)
			agent_action.requires_grad_(True)
			loss = loss_fct(agent_action.to(device), scaled_true_actions.to(device))
			self.pretrain_optimizer.zero_grad()
			loss.backward()
			self.pretrain_optimizer.step()
			index += 1 
			p += done.sum()

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
		next_state, next_obs = self.envs.reset()
		next_done = torch.zeros(self.num_envs).to(device)
		policy_losses = []

		# pretrain...some immitation learning to get us started
		self.pretrain()
		for update in tqdm(range(1, self.num_updates + 1)):
			t0 = time.time()
			# adjust learning rate
			if self.anneal_lr:
				frac = 1.0 - (update - 1.0) / self.num_updates
				lrnow = frac * self.learning_rate
				self.optimizer.param_groups[0]["lr"] = lrnow

			# generate rewards to go for environment before update 
			# They use 20 planner episodes? (Not in steps, but in eps.)
			for step in range(0, self.num_steps):
				global_step += 1 * self.num_envs
				self.buffer.states[step] = next_state
				next_obs = self.normalizeTransition(next_obs)
				self.buffer.observations[step] = next_obs
				self.buffer.terminals[step] = next_done
				next_state, next_obs, next_done = self.rewards_to_go(step, next_state, next_obs, global_step, writer)	
								
			returns, advantages = self.advantages(next_obs, next_done)

			(b_states, b_obs, b_logprobs, b_actions, 
				b_advantages, b_returns, b_values) = self.buffer.flatten(returns, advantages)

			b_inds = np.arange(self.batch_size)
			clip_fracs = []
			for ep in range(self.num_update_epochs):
				# we randomize before processing the minibatches
				np.random.shuffle(b_inds)
				for index in range(0, self.batch_size, self.minibatch_size):
					mb_inds = b_inds[index:index+self.minibatch_size]
					
					_, _, newlogprob, entropy, newvalue = self.policy.evaluate(b_states[mb_inds].to(device), b_obs[mb_inds].to(device), b_actions[mb_inds].to(device))

					# clamp the new and old log probabilities
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
					loss_one = -mb_advantages * ratio
					loss_two = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coeff, 1 + self.clip_coeff)
					policy_loss = torch.max(loss_one, loss_two).mean()
					policy_losses.append(policy_loss.item())

					# value clipping
					newvalue = newvalue.tensor.view(-1)
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