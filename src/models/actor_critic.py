import torch
from torch import nn, tensor
from nets import discrete_net, continuous_net, critic
from torch.distributions import Normal, Categorical
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
# an implementation of the actor-critic 
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
		if(continuous):
			self.actor = continuous_net(hidden_dim, state_dim, action_dim, num_layers, dropout)
			self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_dim)))
		else:
			self.actor = discrete_net(hidden_dim, state_dim, action_dim, num_layers, dropout)
		self.critic = critic(hidden_dim, state_dim, num_layers, dropout) 
		
	def forward(self):
		pass

	# set the new action std. We will also set a new action variance
	def set_action_std(self, new_action_std:float):
		self.action_std_init = new_action_std
		self.action_var = torch.full((self.action_dim, ), self.action_std_init ** 2)

	#@profile
	def act(self, state):
		if(self.continuous):
			action_mean = self.actor(state)
			action_logstd = self.actor_logstd.expand_as(action_mean)
			action_std = torch.exp(action_logstd)
			probs = Normal(action_mean, action_std)
			action = probs.sample()
		else:
			#hidden = self.network(state)
			probs = self.actor(state)
			dist = Categorical(logits=probs)
			action = dist.sample()
		# running the policy to produce values for replay buffer. Can detach.
		return action.detach().cpu(), dist.log_prob(action).detach().cpu(), self.critic(state).detach().cpu()

	def value(self, state):
		return self.critic(state).flatten()

	def evaluate(self, state, action=None):
		if(self.continuous):
			action_mean = self.actor(state)
			action_logstd = self.actor_logstd.expand_as(action_mean)
			dist = Normal(action_mean, action_logstd)
			action = dist.rsample()
		else:
			logits = self.actor(state)
			dist = Categorical(logits=logits)
			if(action is None):
				action = dist.sample()
		#log_prob = dist.log_prob(action).	
		entropy = dist.entropy()
		return action, dist.log_prob(action), dist.entropy(), self.critic(state)