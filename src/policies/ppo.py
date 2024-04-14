from src.policies.policy import policy

class ppo(policy):
    def __init__(self, alpha=1e-2, actor_lr=1e-3, critic_lr=1e-3, alpha_lr=1e-3, gamma=0.99, gae=True, target_update_frequency=1):
        super().__init__()
        self.alpha = alpha
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.gamma = gamma
        self.target_update_frequency = target_update_frequency
        self.tau = 1e-2
        self.gae = gae

    def initNet(self, actor, critic):
        self.pi=actor
        self.critic=critic

        # TODO: Investigate alternative optimization options (grid search?)
        self.pi_optimizer =  torch.optim.Adam(self.pi.parameters(), lr=self.actor_lr)
        self.q_optimizer =  torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.pi_target = deepcopy(self.pi)

    def run_gae(self, next_value, next_done):
		

	# these can be stored in a separate file
	def normal_advantage(self, next_value, next_done):
		# the default rollout scheme
		returns = torch.zeros_like(self.loss_calc_dict['rewards']).to(device)
		for t in reversed(range(self.loss_calc_dict['batch_size'])):
			if t == num_steps - 1:
				nextnonterminal = 1.0 - next_done
				next_return = next_value
			else:
				nextnonterminal = 1.0 - buffer.terminals[t + 1]
				next_return = returns[t + 1]
			returns[t] = self.loss_calc_dict['rewards'][t] + self.gamma * nextnonterminal * next_return
		advantages = returns - self.loss_calc_dict['values']
		return returns, advantages

    # TODO: CHECK TENSOR SHAPING
    def advantages(self, next_obs, next_value, next_done):
		with torch.no_grad():
			next_value = self.critic(next_obs).tensor.flatten()
			if self.gae:
				returns, advantages = self.run_gae(next_value, next_done)
			else:
				returns, advantages = self.normal_advantage(next_value, next_done)
		return returns, advantages
    
    def compute_loss_q(self):
        pass
    
    def compute_loss_pi(self):
        pass

    def update(self, data):
        pass
