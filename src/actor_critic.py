import torch
from torch import nn, tensor
from nets import discrete_net, continuous_net, critic
from torch.distributions import Normal, Categorical

# an implementation of the actor-critic 
class actor_critic(nn.Module):
	def __init__(self, state_dim:int, action_dim:int, 
		hidden_dim:int, num_layers:int, 
		dropout:int, continuous:bool) -> None:
		super().__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.hidden_dim = hidden_dim
		self.continuous = continuous
		self.num_layers = num_layers
		self.dropout = dropout
		if(continuous):
			self.actor = continuous_net(hidden_dim, state_dim, action_dim, num_layers, dropout)
			# the variance of the action space, which we use to sample from the normal distribution
			# this supports std decay from the user
			self.action_var = torch.full((action_dim,), action_std_init ** 2)
			self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_dim)))
		else:
			self.actor = discrete_net(hidden_dim, state_dim, action_dim, num_layers, dropout)
		self.critic = critic(hidden_dim, state_dim, num_layers, dropout) 

	def forward(self):
		pass

	# set the new action std. We will also set a new action variance
	def set_action_std(self, new_action_std:float):
		self.action_std_init = new_action_std
		self.action_var = torch.full((action_dim, ), self.action_std_init ** 2)

	def act(self, state):
		if(self.continuous):
			action_mean = self.actor(stat)
	        action_logstd = self.actor_logstd.expand_as(action_mean)
	        action_std = torch.exp(action_logstd)
	        probs = Normal(action_mean, action_std)
			action = probs.sample()
		else:
			probs = self.actor(state)
			dist = Categorical(probs)
			action = dist.sample()
		log_prob = dist.log_prob(action)	
		state_value = self.critic(state)
		# running the policy to produce values for replay buffer. Can detach.
		return action.detach(), log_prob.detach(), state_value.detach()

	def evaluate(self, state, action):
		if(self.continuous):
			action_mean = self.actor(state)
			action_logstd = self.actor_logstd.expand_as(action_mean)
			dist = Normal(action_mean, action_logstd)
			action = dist.rsample()
		else:
			probs = self.actor(state)
			dist = Categorical(probs)
			action = dist.sample()
		log_prob = dist.log_prob(action)	
		state_value = self.critic(state)
		entropy = dist.entropy()
		return log_prob, state_value, entropy
