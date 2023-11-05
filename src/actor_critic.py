import torch
from torch import nn, tensor
from nets import discrete_net, continuous_net, critic

device = (torch.device('cuda') if torch.cuda_is_available() else torch.device('cpu'))
# an implementation of the actor-critic 
class actor_critic(nn.Module):
	def __init__(self, state_dim:int, action_dim:int, 
		hidden_dim:int, num_layers:int, 
		dropout:int, action_std_init:float, continuous:bool):
		super.__init__(self, actor_critic)
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.hidden_dim = hidden_dim
		self.continuous = continuous
		self.num_layers = num_layers
		self.dropout = dropout
		self.action_std_init = action_std_init
		if(continuous):
			self.actor = continuous_net(hidden_dim, state_dim, action_dim, num_layers, dropout)
			# the variance of the action space, which we use to sample from the normal distribution
			self.action_var = torch.full((action_dim, ), action_std_init ** 2).to(device)
		else:
			self.actor = discrete_net(hidden_dim, state_dim, action_dim, num_layers, dropout)
		self.critic = critic(hidden_dim, state_dim, num_layers, dropout) 

	def forward(self):
		pass

	# set the new action std. We will also set a new action variance
	def set_action_std(self, new_action_std:float):
		self.action_std_init = new_action_std
		self.action_var = torch.full((action_dim, ), self.action_std_init ** 2).to(device)

	def act(self, state):
		if(self.continuous):
			means = self.actor(state)
			cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
			dist = MultivariateNormal(means, cov_mat)
			# rsample vs sample
			action = dist.rsample()
		else:
			probs = self.actor(state)
			dist = Categorical(probs)
			action = dist.sample()
		log_prob = dist.log_prob(action)	
		state_value = self.critic(state)

		# running the policy to produce values for replay buffer. Can detach.
		return action.detach(), log_prob.detach(), state_value.detach()

	def evaluate(self, state, action):
		# for the policy evaluation, we create a multivariate distribution similar
		# to the act method
		if(self.continuous):
			means = self.actor(state)
			cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
			dist = MultivariateNormal(means, cov_mat)
		else:
			probs = self.actor(state)
			dist = Categorical(probs)
			action = dist.sample()
		log_prob = dist.log_prob(action)	
		state_value = self.critic(state)
		entropy = dist.entropy()

		# we don't detach these tensors, because we will use their gradients in the backwards pass
		return log_prob, state_value, entropy
