import torch
from torch import nn, tensor
from nets import discrete_net, continuous_net, critic
from torch.distributions import Normal, Categorical
import numpy as np
import sys

class actor_critic(nn.Module):
	def __init__(self, state_dim:int, action_dim:int, 
		hidden_dim:int, num_layers:int, 
		dropout:int, continuous:bool) -> None:
		super(actor_critic, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.hidden_dim = hidden_dim
		self.continuous = continuous
		self.num_layers = num_layers
		self.dropout = dropout
		# use np.prod to account for multidimensional action space dim inputs
		if(continuous):
			self.actor = continuous_net(hidden_dim, state_dim, action_dim, num_layers, dropout)
			self.critic = critic(hidden_dim, state_dim, num_layers, dropout)
			self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_dim)))
		else:
			self.actor = discrete_net(hidden_dim, state_dim, action_dim, num_layers, dropout)
			self.critic = critic(hidden_dim, state_dim, num_layers, dropout) 
		
	def forward(self):
		pass

	def value(self, state):
		return self.critic(state).flatten()

	def evaluate(self, state, action=None):
		if(self.continuous):
			action_mean = self.actor(state)
			action_logstd = self.actor_logstd.expand_as(action_mean)
			action_std = torch.exp(action_logstd)
			dist = Normal(action_mean, action_std)
			if action is None:
				action = dist.sample()
			log_prob = dist.log_prob(action).sum(1)
			entropy = dist.entropy().sum(1)
		else:
			logits = self.actor(state)
			dist = Categorical(logits=logits)
			if action is None:
				action = dist.sample()
			log_prob = dist.log_prob(action)
			entropy = dist.entropy()
		return action, log_prob, entropy, self.critic(state)
