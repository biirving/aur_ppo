import torch
from torch import nn, tensor
from src.nets.equiv import EquivariantActor, EquivariantCritic
from src.nets.base_cnns import base_actor, base_critic, base_encoder

from torch.distributions import Normal, Categorical
import numpy as np
import sys

epsilon = 1e-6

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
# an implementation of the actor-critic 
class robot_actor_critic(nn.Module):
	def __init__(self, device, 
		equivariant:bool, dx=0.02, dy=0.02, dz=0.02, dr=np.pi/8, n_a=5, tau=0.001) -> None:
		super(robot_actor_critic, self).__init__()
	    # action ranges
		self.p_range = torch.tensor([0, 1])
		self.dtheta_range = torch.tensor([-dr, dr])
		self.dx_range = torch.tensor([-dx, dx])
		self.dy_range = torch.tensor([-dy, dy])
		self.dz_range = torch.tensor([-dz, dz])	
		self.n_a = n_a
		self.device = device
		self.equivariant = equivariant
		# add the equivariant CNN
		if(equivariant):
			self.actor = EquivariantActor().to(device)
			self.critic = EquivariantCritic().to(device)
		else:
			# we could also try to combine the parameters of the actor and the critic
			# I think we should do this. There is nothing inherently wrong with our implementation
			# And I would just have to add a bunch of functionality from here anyways
			# so create a single cnn decoder, and pop two different layers on for each 
			self.network = base_encoder(obs_shape=(2, 128, 128), out_dim=128) 
			#self.actor = nn.Linear(128, 5)
			self.actor = base_actor()
			self.actor.apply(weights_init)
			self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(5)))
			#self.critic = torch.nn.Sequential(
			#	torch.nn.Linear(128, 128),
			#	nn.ReLU(inplace=True),
			#	torch.nn.Linear(128, 1)
			#)
			self.critic = base_critic()
			self.critic.apply(weights_init)
		
	def forward(self, act):
		pass

	def value(self, state, obs):
		state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
		cat_obs = torch.cat([obs, state_tile], dim=1).to(self.device)
		return self.critic(cat_obs)

	# courtesy of Dian Wang
	def decodeActions(self, *args):
		unscaled_p = args[0]
		unscaled_dx = args[1]
		unscaled_dy = args[2]
		unscaled_dz = args[3]

		p = 0.5 * (unscaled_p + 1) * (self.p_range[1] - self.p_range[0]) + self.p_range[0]
		dx = 0.5 * (unscaled_dx + 1) * (self.dx_range[1] - self.dx_range[0]) + self.dx_range[0]
		dy = 0.5 * (unscaled_dy + 1) * (self.dy_range[1] - self.dy_range[0]) + self.dy_range[0]
		dz = 0.5 * (unscaled_dz + 1) * (self.dz_range[1] - self.dz_range[0]) + self.dz_range[0]

		if self.n_a == 5:
			unscaled_dtheta = args[4]
			dtheta = 0.5 * (unscaled_dtheta + 1) * (self.dtheta_range[1] - self.dtheta_range[0]) + self.dtheta_range[0]
			actions = torch.stack([p, dx, dy, dz, dtheta], dim=1)
			unscaled_actions = torch.stack([unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz, unscaled_dtheta], dim=1)
		else:
			actions = torch.stack([p, dx, dy, dz], dim=1)
			unscaled_actions = torch.stack([unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz], dim=1)
		return unscaled_actions, actions

	# scaled actions
	def getActionFromPlan(self, plan):
		def getUnscaledAction(action, action_range):
			unscaled_action = 2 * (action - action_range[0]) / (action_range[1] - action_range[0]) - 1
			return unscaled_action
		dx = plan[:, 1].clamp(*self.dx_range)
		p = plan[:, 0].clamp(*self.p_range)
		dy = plan[:, 2].clamp(*self.dy_range)
		dz = plan[:, 3].clamp(*self.dz_range)
		unscaled_p = getUnscaledAction(p, self.p_range)
		unscaled_dx = getUnscaledAction(dx, self.dx_range)
		unscaled_dy = getUnscaledAction(dy, self.dy_range)
		unscaled_dz = getUnscaledAction(dz, self.dz_range)
		if self.n_a == 5:
			dtheta = plan[:, 4].clamp(*self.dtheta_range)
			unscaled_dtheta = getUnscaledAction(dtheta, self.dtheta_range)
			return self.decodeActions(unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz, unscaled_dtheta)
		else:
			return self.decodeActions(unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz)

	def evaluate(self, state, obs, action=None):
		# so the state is whether the gripper is open or closed
		state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
		cat_obs = torch.cat([obs, state_tile], dim=1).to(self.device)
		
		if(self.equivariant):
			action_mean, action_logstd = self.actor(cat_obs)
		else:
			action_mean = self.actor(cat_obs)
			action_logstd = self.actor_logstd.expand_as(action_mean)

		action_std = torch.exp(action_logstd)
		dist = Normal(action_mean, action_std)

		if action is None:
			action = dist.rsample()

		#action = torch.tanh(action)
		log_prob = dist.log_prob(action)

		# clipping the log probability
		#log_prob -= torch.log((1 - action.pow(2)) + epsilon)
		#log_prob = log_prob.sum(1, keepdim=True)

		#action_mean = torch.tanh(action_mean)
		entropy = dist.entropy()		
		unscaled_actions, actions = self.decodeActions(*[action[:, i] for i in range(self.n_a)])
		return actions, unscaled_actions, log_prob.sum(1), entropy.sum(1), self.critic(cat_obs)

	# pretrain the actor alone
	def evaluate_pretrain(self, state, obs, action=None):
		state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
		cat_obs = torch.cat([obs, state_tile], dim=1).to(self.device)
		if(self.equivariant):
			action_mean, action_logstd = self.actor(cat_obs)
		else:
			action_mean = self.actor(self.network(cat_obs))
			action_logstd = self.actor_logstd.expand_as(action_mean)
		action_std = torch.exp(action_logstd)
		dist = Normal(action_mean, action_std)
		if action is None:
			action = dist.rsample()
		action = torch.tanh(action)
		action_mean = torch.tanh(action_mean)
		unscaled_action, action = self.decodeActions(*[action[:, i] for i in range(self.n_a)])
		return action.to(torch.float16), unscaled_action.to(torch.float16)


	def test_action(self, state, obs):
		state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
		cat_obs = torch.cat([obs, state_tile], dim=1).to(self.device)
		action_mean = self.actor(cat_obs)
		mean = torch.tanh(action_mean)
		return self.decodeActions(*[mean[:, i] for i in range(self.n_a)])
