import torch
from torch import nn, tensor
from nets.equiv import EquivariantActor, EquivariantCritic
from torch.distributions import Normal, Categorical
import numpy as np
import sys

# an implementation of the actor-critic 
class robot_actor_critic(nn.Module):
	def __init__(self, device, 
		equivariant:bool, dx=0.005, dy=0.005, dz=0.005, dr=np.pi/16, n_a=5, tau=0.001) -> None:
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
			#self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_dim)))
		
	def forward(self, act):
		pass

	def value(self, state):
		return self.critic(state)

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
		# support batching
		state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
		cat_obs = torch.cat([obs, state_tile], dim=1).to(self.device)
		action_mean, action_logstd = self.actor(cat_obs)
		action_std = torch.exp(action_logstd)
		dist = Normal(action_mean, action_std)
		if action is None:
			action = dist.sample()
		log_prob = dist.log_prob(action)
		entropy = dist.entropy()
		unscaled_actions, actions = self.decodeActions(*[action[:, i] for i in range(self.n_a)])
		return actions, unscaled_actions, log_prob.sum(1), entropy.sum(1), self.critic(obs)
