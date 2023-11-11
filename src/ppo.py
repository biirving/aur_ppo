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
class ppo():
	def __init__(self, params):
		self.gym_id = params['gym_id']
		self.exp_name = params['exp_name']
		self.seed = params['seed']
		self.num_steps = params['num_steps']
		self.gae = params['gae']
		self.total_timesteps = params['total_timesteps']
		self.anneal_lr = params['anneal_lr']
		self.gae_lambda = params['gae_lambda']
		self.num_update_epochs = params['num_update_epochs']
		self.num_envs = params['num_envs']
		self.num_minibatches = params['num_minibatches']
		self.batch_size = self.num_steps * self.num_envs
		self.minibatch_size = int(self.batch_size // self.num_minibatches)
		self.entropy_coeff = params['entropy_coeff']
		self.value_coeff = params['value_coeff']
		self.clip_coef = params['clip_coeff']
		self.max_grad_norm = params['max_grad_norm']
		self.target_kl = params['target_kl']
		self.norm_adv = params['norm_adv']
		self.hidden_dim = params['hidden_dim']
		self.continuous = params['continuous']
		self.learning_rate = params['learning_rate']
		self.num_layers = params['num_layers']
		self.dropout = params['dropout']
		self.gamma = params['gamma']
		self.num_updates = self.total_timesteps // self.batch_size
		run_name = f"{self.gym_id}__{self.exp_name}__{self.seed}__{int(time.time())}"
		self.env = gym.vector.SyncVectorEnv(
	    	[self.make_env(self.gym_id, int(self.seed) + i, i, params['capture_video'], run_name) for i in range(self.num_envs)]
	    )
		self.state_dim = self.env.single_observation_space.shape
		if(self.continuous):
			self.action_dim = self.env.single_action_space.shape
		else:
			self.action_dim = self.env.single_action_space.n
		#print(self.env.single_action_space.shape)	
		self.policy = actor_critic(self.state_dim[0], 
			self.action_dim, self.hidden_dim, self.num_layers, self.dropout, self.continuous)
		self.policy_old = actor_critic(self.state_dim[0], 
			self.action_dim, self.hidden_dim, self.num_layers, self.dropout, self.continuous)
		self.policy_old.load_state_dict(self.policy.state_dict())


		self.buffer = torch_buffer(self.state_dim, self.env.single_action_space.shape, self.num_steps, self.num_envs)
		self.optimizer =  torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate, eps=1e-5)


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
	def rewards_to_go(self, step, next_obs):
		with torch.no_grad():
			action, logprob, value = self.policy_old.act(next_obs)
			self.buffer.values[step] = value.flatten()
			self.buffer.actions[step] = action
			self.buffer.log_probs[step] = logprob
			next_obs, reward, done, _ = self.env.step(action.cpu().numpy())
			self.buffer.rewards[step] = torch.tensor(reward).to(device).view(-1)
			next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
		return next_obs, next_done

	def advantages(self, next_obs, next_done):
		with torch.no_grad():
			next_value = self.policy_old.critic.forward(next_obs).reshape(1, -1)
            # generalized advantage estimation
			if self.gae:
				advantages = torch.zeros_like(self.buffer.rewards).to(device)
				lastgaelam = 0
				for t in reversed(range(self.num_steps)):
					if t == self.num_steps - 1:
						nextnonterminal = 1.0 - next_done
                        # boot strap from the last returned value, not
                        # stored in the buffer
						nextvalues = next_value
					else:
						nextnonterminal = 1.0 - self.buffer.terminals[t + 1]
						nextvalues = self.buffer.values[t + 1]
					delta = self.buffer.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.buffer.values[t]
					advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
				returns = advantages + self.buffer.values
			else:
				returns = torch.zeros_like(self.rewards).to(device)
				for t in reversed(range(self.num_steps)):
					if t == self.num_steps - 1:
						nextnonterminal = 1.0 - next_done
						next_return = next_value
					else:
						nextnonterminal = 1.0 - self.terminals[t + 1]
						next_return = returns[t + 1]
					returns[t] = self.buffer.rewards[t] + self.gamma * nextnonterminal * next_return
				advantages = returns - self.buffer.values
		return returns, advantages

	def plot(self, loss):
		print(loss.shape)
		timesteps = np.arange(1, loss.shape[0] + 1)
		plt.plot(timesteps, loss)
		plt.xlabel('Timestep')
		plt.ylabel('Episode length')
		plt.title('Episode Length over time')
		plt.show()

	def train(self):
		global_step = 0
		start_time = time.time()
		next_obs = torch.Tensor(self.env.reset()).to(device)
		next_done = torch.zeros(self.num_envs).to(device)
		policy_losses = []
		for update in tqdm(range(1, self.num_updates + 1)):
			if self.anneal_lr:
				frac = 1.0 - (update - 1.0) / self.num_updates
				lrnow = frac * self.learning_rate
				self.optimizer.param_groups[0]["lr"] = lrnow
			for step in range(0, self.num_steps):
				global_step += 1 * self.num_envs
				self.buffer.states[step] = next_obs
				self.buffer.terminals[step] = next_done
				next_obs, next_done = self.rewards_to_go(step, next_obs)
			returns, advantages = self.advantages(next_obs, next_done)
			# we flatten here to have access to all of the environment returns
			b_obs = self.buffer.states.reshape((-1,) + self.env.single_observation_space.shape)
			b_logprobs = self.buffer.log_probs.reshape(-1)
			b_actions = self.buffer.actions.reshape((-1,) + self.env.single_action_space.shape)
			b_advantages = advantages.reshape(-1)
			b_returns = returns.reshape(-1)
			b_values = self.buffer.values.reshape(-1)
			b_inds = np.arange(self.batch_size)
			clip_fracs = []
			for ep in range(self.num_update_epochs):
				np.random.shuffle(b_inds)
				for index in range(0, self.batch_size, self.minibatch_size):
					mb_inds = b_inds[index:index+self.minibatch_size]
					log_prob, state_value, entropy = self.policy.evaluate(b_obs[mb_inds], b_actions[mb_inds])
					log_ratio = log_prob - b_logprobs[mb_inds]
					ratio = log_ratio.exp()
					# to check for early stopping. should remain below 0.2
					with torch.no_grad():
						# http://joschu.net/blog/kl-approx.html
						old_approx_kl = (-log_ratio).mean()
						approx_kl = ((ratio - 1) - log_ratio).mean()
						clip_fracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]
					# we normalize at the minibatch level
					mb_advantages = b_advantages[mb_inds]
					# 1e-8 avoids division by 0
					if self.norm_adv:
						mb_advantages = (mb_advantages - mb_advantages.mean())/(mb_advantages.std() + 1e-8)
					# gradient descent, rather than ascent
					loss_one = -mb_advantages * ratio
					loss_two = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
					policy_loss = torch.max(loss_one, loss_two).mean()

					# mean squared error
					value_loss = 0.5 * (((state_value - b_values)) ** 2).mean()
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
			# fix debug metrics           	
			policy_losses.append(policy_loss.item())
		self.plot(np.array(policy_losses))

