import torch
from torch import tensor, nn
import actor_critic




# proximal policy optimization
class ppo(nn.Module):
	def __init__(self, gamma:float, alpha:float, 
		epsilon:float, env,  state_dim:int, state_dim:int, hidden_dim:int, 
		num_layers:int, dropout:float):
		self.gamma = gamma
		self.alpha = alpha
		self.hidden_dim = dim
		self.layers = layers
		self.dropout = dropout
		self.action_dim = action_dim
		self.state_dim = state_dim
		self.epsilon = epsilon

		# tentative starting std
		self.action_std_init = 0.6
		self.actor_critic = actor_critic(state_dim, 
			action_dim, 
			hidden_dim, 
			num_layers, 
			dropout, 
			self.action_std_init)

		# we optimize both the actor and the critic
		self.optimizer = torch.optim.Adam( 
			{'params':self.actor_critic.actor.parameters()},
			{'params':self.actor_critic.critic.parameters()}
			)
		
	def forward(self, input):










