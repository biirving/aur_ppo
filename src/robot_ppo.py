import torch
from torch import nn, tensor
from nets import discrete_net, continuous_net, critic
from torch.distributions import MultivariateNormal, Categorical
import gym
import time
from models import robot_actor_critic
import numpy as np
import matplotlib.pyplot as plt
import sys, os, time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
from env_wrapper import EnvWrapper
from torch.distributions import Normal, Categorical
import collections

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class torch_buffer():
	def __init__(self, state_shape, observation_shape, action_shape, num_steps, num_envs):
		self.state_shape = state_shape
		self.observation_shape = observation_shape
		self.action_shape = action_shape
		self.states = torch.zeros((num_steps, num_envs))
		self.observations = torch.zeros((num_steps, num_envs) + observation_shape)
		self.actions = torch.zeros((num_steps, num_envs, action_shape))
		# to store the actions from the expert rollout
		self.true_actions = torch.zeros((num_steps, num_envs, action_shape))
		self.log_probs = torch.zeros((num_steps, num_envs, action_shape))
		self.rewards = torch.zeros((num_steps, num_envs))
		self.terminals = torch.zeros((num_steps, num_envs))
		self.values = torch.zeros((num_steps, num_envs))

	def load_to_device(self):
		self.states = self.states.to(device)
		self.observations = self.observations.to(device)
		self.actions = self.actions.to(device)
		self.log_probs = self.log_probs.to(device)
		self.rewards = self.rewards.to(device)
		self.terminals = self.terminals.to(device)
		self.values = self.values.to(device)
		self.true_actions = self.true_actions.to(device)

	def load_to_cpu(self):
		self.states = self.states.to('cpu')
		self.observations = self.observations.to('cpu')
		self.actions = self.actions.to('cpu')
		self.log_probs = self.log_probs.to('cpu')
		self.rewards = self.rewards.to('cpu')
		self.terminals = self.terminals.to('cpu')
		self.values = self.values.to('cpu')
		self.true_actions.to('cpu')
	
	def flatten(self, returns, advantages):
		b_states = self.states.view(self.states.shape[0] * self.states.shape[1])
		b_obs = self.observations.view(self.observations.shape[0] * self.observations.shape[1], 
								 self.observations.shape[2], self.observations.shape[3], self.observations.shape[4])
		b_logprobs = self.log_probs.reshape(-1)
		b_actions = self.actions.view(self.actions.shape[0] * self.actions.shape[1], self.actions.shape[2])
		b_true_actions = self.true_actions.view(self.true_actions.shape[0] * self.true_actions.shape[1], self.true_actions.shape[2])
		b_advantages = advantages.reshape(-1)
		b_returns = returns.reshape(-1)
		b_values = self.values.reshape(-1)
		return (b_states, b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, b_true_actions)

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
	def __init__(self, params):
		# accessing through the dictionary is slower
		self.params_dict = params
		self.all_steps = None
		self.minibatch_size = None
		# Define a list of attributes to exclude from direct assignment
		exclude_list = ['batch_size', 'minibatch_size']
		# Iterate through the keys in params
		for key, value in params.items():
			if key not in exclude_list:
				setattr(self, key, value)

		# Tht total steps x the number of envs represents how many total
		# steps in the said environment will be taken by the training loop		
		self.all_steps = self.num_steps * self.num_envs 
		self.batch_size = int(self.num_envs * self.num_steps)
		self.minibatch_size = int(self.all_steps // self.num_minibatches)
		self.num_updates = self.total_timesteps // self.batch_size
		self.run_name = f"{self.gym_id}__{self.exp_name}__{self.seed}__{int(time.time())}"

		self.total_pretrain_steps = self.pretrain_steps * self.num_envs
		self.pretrain_minibatch_size = int(self.total_pretrain_steps // self.pretrain_batch_size)


		num_eval_processes=5
		simulator='pybullet'
		env=self.gym_id
		# config used in Dian, Rob, and Robin's paper
		env_config={'workspace': np.array([[ 0.25,  0.65],
			[-0.2 ,  0.2 ],
			[ 0.01,  0.25]]), 'max_steps': 250, 
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

		self.action_dim = 5
		self.state_dim = 1 
		self.buffer = torch_buffer(self.state_dim, (1, 128, 128), self.action_dim, self.num_steps, self.num_envs)
		# move the primary buffer onto the device
		self.buffer.load_to_device()
		# Have a set number of pretrain steps

		self.pretrain_buffer = torch_buffer(self.state_dim, (1, 128, 128), self.action_dim, self.pretrain_steps, self.num_envs)

		self.optimizer =  torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate, eps=1e-5)
		self.pretrain_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=self.learning_rate, eps=1e-5)
		self.scaler = torch.cuda.amp.GradScaler()
		self.total_returns = []
		self.total_episode_lengths = []
		self.x_indices = []
		self.episodic_returns = store_returns(self.num_envs, self.gamma)

	def rewards_to_go(self, step, next_state, next_obs, global_step, writer):
		with torch.no_grad():
			actions, unscaled, logprob, _, value = self.policy.evaluate(next_state.to(device), next_obs.to(device))
			if(self.equivariant):
				self.buffer.values[step] = value.tensor.flatten()
			else:
				self.buffer.values[step] = value.flatten()
		self.buffer.actions[step] = unscaled 
		self.buffer.log_probs[step] = logprob
		self.buffer.true_actions[step] = self.envs.getNextAction()

		#self.buffer.true_actions[step] = true_actions
		next_states, next_obs, reward, done = self.envs.step(actions, auto_reset=True)
		for i, rew in enumerate(reward):
			self.episodic_returns.add_value(i, rew)

		self.buffer.rewards[step] = reward.view(-1)
		next_states, next_obs, next_done = next_states.to(device), next_obs.to(device), done.to(device)
		for i, d in enumerate(done):
			if d:
				discounted_return, episode_length = self.episodic_returns.calc_discounted_return(i)
				writer.add_scalar("charts/discounted_episodic_return", discounted_return, global_step)
				writer.add_scalar("charts/episodic_length", episode_length, global_step)
				break
		return next_states, next_obs, next_done


	# behaviour cloning?
	def expert_rollout(self, step, state, obs):
		with torch.no_grad():
			true_action = self.envs.getNextAction()
			unscaled, scaled = self.policy.getActionFromPlan(true_action)
			_, unscaled_agent, logprob, _, value = self.policy.evaluate(state.to(device), obs.to(device))
			# to store for mse loss
			self.pretrain_buffer.actions[step] = unscaled_agent

			if(self.equivariant):
				self.pretrain_buffer.values[step] = value.tensor.detach().flatten()
			else:
				self.pretrain_buffer.values[step] = value.detach().flatten()

		self.pretrain_buffer.true_actions[step] = unscaled 
		self.pretrain_buffer.log_probs[step] = logprob
		next_states, next_obs, reward, done = self.envs.step(scaled, auto_reset=True)
		for i, rew in enumerate(reward):
			self.episodic_returns.add_value(i, rew)
		self.pretrain_buffer.rewards[step] = reward.view(-1)
		next_states, next_obs, next_done = next_states, next_obs, done
		return next_states, next_obs, next_done

	def run_gae(self, next_value, next_done, buffer, num_steps):
		advantages = torch.zeros_like(buffer.rewards)
		lastgaelam = 0
		for t in reversed(range(num_steps - 1)):
			if t == self.num_steps - 1:
				nextnonterminal = 1.0 - next_done
				nextvalues = next_value
			else:
				nextnonterminal = 1.0 - buffer.terminals[t + 1]
				nextvalues = buffer.values[t + 1]
			# If episode finishes after timestep t, we mask the value and the previous advantage value, 
			# so that advantages[t] stores the (reward - value) at that timestep without
			# taking a value from the next episode, or storing a value in this episode to be multiplied
			# in the calculation of the rollout of the next episode
			delta = buffer.rewards[t] + self.gamma * nextvalues * nextnonterminal - buffer.values[t]
			advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
		returns = advantages + buffer.values
		return returns, advantages

	# these can be stored in a separate file
	def normal_advantage(self, next_value, next_done, buffer, num_steps):
		# the default rollout scheme
		returns = torch.zeros_like(buffer.rewards).to(device)
		for t in reversed(range(num_steps)):
			if t == num_steps - 1:
				nextnonterminal = 1.0 - next_done
				next_return = next_value
			else:
				nextnonterminal = 1.0 - buffer.terminals[t + 1]
				next_return = returns[t + 1]
			returns[t] = buffer.rewards[t] + self.gamma * nextnonterminal * next_return
		advantages = returns - buffer.values
		return returns, advantages

	def advantages(self, next_state, next_obs, next_done, buffer, num_steps):
		with torch.no_grad():
			if(self.equivariant):
				next_value = self.policy.value(next_state.to(device), next_obs.to(device)).tensor.flatten()
			else:
				next_value = self.policy.value(next_state.to(device), next_obs.to(device)).flatten()
			if self.gae:
				returns, advantages = self.run_gae(next_value, next_done, buffer, num_steps)
			else:
				returns, advantages = self.normal_advantage(next_value, next_done, buffer, num_steps)
		return returns, advantages

	# Fill out the pretrain buffer 
	def pretrain(self):
		"""
		Simple behavioral cloning
		"""
		state, obs = self.envs.reset()
		done = torch.zeros(self.num_envs).to(device)
		for step in tqdm(range(0, self.pretrain_steps)):
			self.pretrain_buffer.states[step] = state
			self.pretrain_buffer.observations[step] = obs
			self.pretrain_buffer.terminals[step] = done 
			state, obs, done = self.expert_rollout(step, state, obs)
		return state.to(device), obs.to(device), done.to(device)

	def update(self, buffer, update_epochs, batch_size, minibatch_size, policy_losses, pretrain=False):
		(b_states, b_obs, b_logprobs, b_actions, 
			b_advantages, b_returns, b_values, b_true_actions) = buffer 

		b_inds = np.arange(batch_size)
		clip_fracs = []
		for ep in range(update_epochs):
			np.random.shuffle(b_inds)
			for index in range(0, batch_size, minibatch_size):
				mb_inds = b_inds[index:index+self.minibatch_size]
				
				_, _, newlogprob, entropy, newvalue = self.policy.evaluate(b_states[mb_inds].to(device), b_obs[mb_inds].to(device), b_actions[mb_inds].to(device))

				old_log_probs = b_logprobs[mb_inds]
				log_ratio = newlogprob.to(device) - old_log_probs.to(device)

				ratio = log_ratio.exp()

				with torch.no_grad():	
					old_approx_kl = (-log_ratio).mean()
					approx_kl = ((ratio - 1) - log_ratio).mean()
					clip_fracs += [((ratio - 1.0).abs() > self.clip_coeff).float().mean().item()]

				# we normalize at the minibatch level
				mb_advantages = b_advantages[mb_inds]

				# 1e-8 avoids division by 0
				if self.norm_adv:
					mb_advantages = (mb_advantages - mb_advantages.mean())/(mb_advantages.std() + 1e-8)

				loss_one = -mb_advantages * ratio
			
				loss_two = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coeff, 1 + self.clip_coeff)
				policy_loss = torch.max(loss_one, loss_two).mean()
				policy_losses.append(policy_loss.item())

				# value clipping
				if(self.equivariant):
					newvalue = newvalue.tensor.view(-1)
				else:
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

				# Data Aggregation 
				entropy_loss = entropy.mean()
				if(pretrain):
					# the simple mse loss between the proposed actions from the arm and the "true" actions
					expert_loss = nn.functional.mse_loss(b_actions[mb_inds].requires_grad_(True), b_true_actions[mb_inds])
					#loss = policy_loss - self.entropy_coeff * entropy_loss + value_loss * self.value_coeff + self.expert_weight * expert_loss
					loss = self.expert_weight * expert_loss
				else:
					# also, should I have the expert weight and loss throughout entire training loop?
					loss = policy_loss - self.entropy_coeff * entropy_loss + value_loss * self.value_coeff	

				self.optimizer.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
				self.optimizer.step()
			if self.target_kl is not None:
				if approx_kl > self.target_kl:
					break
		return (policy_loss, value_loss, entropy_loss, old_approx_kl, approx_kl, clip_fracs)



	#@profile
	def train(self):
		if self.track:
			import wandb
			wandb.init(project='ppo',entity='Aurelian',sync_tensorboard=True,config=None,name=self.gym_id + '_' + str(self.learning_rate) + '_' + 
			str(self.value_coeff) + '_' + str(self.entropy_coeff) + '_' + str(self.clip_coeff) + '_' + str(self.num_minibatches),monitor_gym=True,save_code=True)
		writer = SummaryWriter(f"runs/{self.gym_id}")
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


		print('Pretraining...')
		self.policy.train()
		# Populate the pretrain buffer
		pretrain_policy_losses = []
		next_state, next_obs, next_done = self.pretrain()
		self.pretrain_buffer.load_to_device()
		returns, advantages = self.advantages(next_state, next_obs, next_done, self.pretrain_buffer, self.pretrain_steps)
		returns = returns.to(device)
		advantages = advantages.to(device)
		# load our pretrain buffer onto the device
		flattened_pretrain_buffer = self.pretrain_buffer.flatten(returns, advantages)
		# update the weights of the network based on the results of the pretrain buffer
		self.update(flattened_pretrain_buffer, self.num_update_epochs, self.pretrain_batch_size, self.pretrain_minibatch_size, pretrain_policy_losses, pretrain=True)

		# then offload the pretrain buffer from the device
		self.pretrain_buffer.load_to_cpu()
		print('Pretraining complete')

		global_step = 0
		start_time = time.time()
		next_state, next_obs = self.envs.reset()
		next_done = torch.zeros(self.num_envs).to(device)
		policy_losses = []

		for update in tqdm(range(1, self.num_updates + 1)):
			t0 = time.time()
			# adjust learning rate
			if self.anneal_lr:
				frac = 1.0 - (update - 1.0) / self.num_updates
				lrnow = frac * self.learning_rate
				self.optimizer.param_groups[0]["lr"] = lrnow

			# we want to anneal the expert weight as well
			if self.anneal_exp:
				frac = 1 - ((update - 1)/self.num_updates)
				self.expert_weight *= frac

			# For some portion of the episodes, we could have the expert rollout the 'expert' actions
			for step in tqdm(range(0, self.num_steps)):
				global_step += 1 * self.num_envs
				self.buffer.states[step] = next_state
				self.buffer.observations[step] = next_obs
				self.buffer.terminals[step] = next_done
				next_state, next_obs, next_done = self.rewards_to_go(step, next_state, next_obs, global_step, writer)	

			# should we do expert rollout in the same manner?	
			returns, advantages = self.advantages(next_state, next_obs, next_done, self.buffer, self.num_steps)
			buffer = self.buffer.flatten(returns, advantages)

			(policy_loss, 
			value_loss, 
			entropy_loss, 
			old_approx_kl, 
			approx_kl, 
			clip_fracs) = self.update(buffer, self.num_update_epochs, self.batch_size, self.minibatch_size, policy_losses)	

			policy_losses.append(policy_loss.item())

			y_pred, y_true = buffer[6].cpu().numpy(), buffer[5].cpu().numpy()
			var_y = np.var(y_true)
			explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


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
		if(self.save_file_path is not None):
			# save the dictionary states
			save_state = {'actor_state':self.policy.actor.state_dict(),
					'critic_state':self.policy.critic.state_dict(), 
					'optimizer_state':self.optimizer.state_dict()}
			torch.save(save_state, self.save_file_path  + 'actor_critic_' + str(self.num_layers) + '.pt')
		#self.plot_episodic_returns(np.array(self.total_returns), np.array(np.array(self.x_indices)), 'episodic returns')
		#self.plot_episodic_returns(np.array(self.total_episode_lengths), np.array(np.array(self.x_indices)), 'episodic lengths')

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