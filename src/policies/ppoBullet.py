from src.bulletArmPolicy import bulletArmPolicy

class ppoBullet(bulletArmPolicy):
    def __init__(self, alpha=1e-2, actor_lr=1e-3, critic_lr=1e-3, alpha_lr=1e-3, 
    gamma=0.99, gae=True, target_update_frequency=1, num_processes=5, 
    total_steps=10000, update_epochs=10, num_minibatches=50):
        super().__init__()
        self.alpha = alpha
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.gamma = gamma
        self.target_update_frequency = target_update_frequency
        self.tau = 1e-2

        # TODO: Add these to abstract PPO class
        self.gae = gae
        self.minibatch_size = int(num_processes * total_steps) // num_minibatches
        self.num_update_epochs = update_epochs

    def initNet(self, actor, critic):
        self.pi=actor
        self.critic=critic

        # TODO: Investigate alternative optimization options (grid search?)
        self.pi_optimizer =  torch.optim.Adam(self.pi.parameters(), lr=self.actor_lr)
        self.q_optimizer =  torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.pi_target = deepcopy(self.pi)

    def _loadBatchToDevice(self, batch, device='cuda'):
        """
        Load batch into pytorch tensor. The functionality of this method varies for on-policy and off policy
        methods.

        :param batch: list of transitions
        :return: states_tensor, obs_tensor, action_tensor, rewards_tensor, next_states_tensor, next_obs_tensor,
                 non_final_masks, step_lefts_tensor, is_experts_tensor
        
        """
        states = []
        images = []
        xys = []
        rewards = []
        dones = []
        step_lefts = []
        values = []
        expert_actions = []
        log_probs = []

        for d in batch:
            states.append(d.state)
            images.append(d.obs)
            xys.append(d.action)
            rewards.append(d.reward.squeeze())
            dones.append(d.done)
            step_lefts.append(d.step_left)
            values.append(d.value)
            expert_actions.append(d.expert_action)
            log_probs.append(d.log_prob)

        states_tensor = torch.tensor(np.stack(states)).long().to(device)
        obs_tensor = torch.tensor(np.stack(images)).to(device)
        if len(obs_tensor.shape) == 3:
            obs_tensor = obs_tensor.unsqueeze(1)
        action_tensor = torch.tensor(np.stack(xys)).to(device)
        rewards_tensor = torch.tensor(np.stack(rewards)).to(device)
        dones_tensor = torch.tensor(np.stack(dones)).int()
        non_final_masks = (dones_tensor ^ 1).float().to(device)
        step_lefts_tensor = torch.tensor(np.stack(step_lefts)).to(device)
        values_tensor = torch.tensor(np.stack(values)).to(device)
        expert_actions_tensor = torch.tensor(np.stack(true_actions)).to(device)
        log_probs_tensor = torch.tensor(np.stack(log_probs)).to(device)

        # scale observation from int to float
        obs_tensor = obs_tensor/255*0.4

        self.loss_calc_dict['batch_size'] = len(batch)
        self.loss_calc_dict['states'] = states_tensor
        self.loss_calc_dict['obs'] = obs_tensor
        self.loss_calc_dict['action_idx'] = action_tensor
        self.loss_calc_dict['rewards'] = rewards_tensor
        self.loss_calc_dict['non_final_masks'] = non_final_masks
        self.loss_calc_dict['step_lefts'] = step_lefts_tensor
        self.loss_calc_dict['values'] = values_tensor
        self.loss_calc_dict['expert_actions'] = expert_actions_tensor
        self.loss_calc_dict['log_probs'] = log_probs_tensor
        
        return states_tensor, obs_tensor, action_tensor, rewards_tensor, \
                non_final_masks, step_lefts_tensor, values_tensor, expert_actions_tensor, log_probs_tensor
    def load_info(self):
        batch_size = self.loss_calc_dict['batch_size']
        states = self.loss_calc_dict['states']
        obs = self.loss_calc_dict['obs']
        action_idx = self.loss_calc_dict['action_idx']
        rewards = self.loss_calc_dict['rewards']
        non_final_masks = self.loss_calc_dict['non_final_masks']
        step_lefts = self.loss_calc_dict['step_lefts']
        values = self.loss_calc_dict['values']
        expert_actions = self.loss_calc_dict['expert_actions']
        log_probs = self.loss_calc_dict['log_probs']
        return batch_size, states, obs, action_idx, rewards, non_final_masks, step_lefts, values, expert_actions, log_probs

    def _loadLossCalcDict(self):
        """
        get the loaded batch data in self.loss_calc_dict
        :return: batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts
        """
        batch_size, states, obs, action_idx, rewards, non_final_masks, step_lefts, values, expert_actions log_probs= self.load_info() 

        if self.obs_type is 'pixel':
            # stack state as the second channel of the obs
            obs = torch.cat([obs, states.reshape(states.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])], dim=1)
        return batch_size, states, obs, action_idx, rewards, non_final_masks, step_lefts, values, expert_actions, log_probs


    def run_gae(self, next_value, next_done):
		advantages = torch.zeros_like(self.loss_calc_dict['rewards'])
		lastgaelam = 0
        batch_size, states, obs, actions, rewards, dones, steps_left, values, expert, log_probs = self._loadLossCalcDict()
		for t in reversed(range(batch_size)):
			if t == self.num_steps - 1:
				nextnonterminal = 1.0 - next_done
				nextvalues = next_value
			else:
				nextnonterminal = 1.0 - dones[t + 1]
				nextvalues = values[t + 1]
			# If episode finishes after timestep t, we mask the value and the previous advantage value, 
			# so that advantages[t] stores the (reward - value) at that timestep without
			# taking a value from the next episode, or storing a value in this episode to be multiplied
			# in the calculation of the rollout of the next episode
			delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
			advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
		returns = advantages + values
		return returns, advantages

	# these can be stored in a separate file
	def normal_advantage(self, next_value, next_done):
        batch_size, states, obs, actions, rewards, dones, steps_left, values, expert, log_probs = self._loadLossCalcDict()
		# the default rollout scheme
		returns = torch.zeros_like(rewards).to(device)
		for t in reversed(range(batch_size)):
			if t == num_steps - 1:
				nextnonterminal = 1.0 - next_done
				next_return = next_value
			else:
				nextnonterminal = 1.0 - buffer.terminals[t + 1]
				next_return = returns[t + 1]
			returns[t] = rewards[t] + self.gamma * nextnonterminal * next_return
		advantages = returns - values
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

    def update(self, data, next_obs, next_value, next_done):
        self._loadBatchToDevice(data)

		returns, advantages = self.advantages(next_obs, next_value, next_done)
        # then assuming we have some reshaping in here
        for uep in self.num_update_epochs:
            b_inds = np.arange(batch_size)
            for index in range(0, batch_size, self.minibatch_size):
				mb_inds = b_inds[index:index+self.minibatch_size]
				_, _, newlogprob, entropy, newvalue = self.policy.evaluate(b_states[mb_inds].to(device), b_obs[mb_inds].to(device), b_actions[mb_inds].to(device))
				old_log_probs = b_logprobs[mb_inds]
				log_ratio = newlogprob.to(device) - old_log_probs.to(device)
				ratio = log_ratio.exp()
				with torch.no_grad():	
					old_approx_kl = (-log_ratio).mean()
					approx_kl = ((ratio - 1) - log_ratio).mean()
					clip_fracs += [((ratio - 1.0).abs() > self.clip_coeff).float().mean().item()]
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

				entropy_loss = entropy.mean()
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
				value_loss *= self.value_coeff

				expert_loss = nn.functional.mse_loss(b_actions[mb_inds], b_true_actions[mb_inds])
				loss = policy_loss - self.entropy_coeff * entropy_loss + value_loss + self.expert_weight * expert_loss

				loss.backward()
				nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
				self.optimizer.step()
				self.optimizer.zero_grad()

        qf1_loss, qf2_loss  = self.compute_loss_q()
        qf_loss = qf1_loss + qf2_loss
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        loss_pi = self.compute_loss_pi()
        log_pi = self.loss_calc_dict['log_pi']
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        # use the tau copying mechanism?
        tau = self.tau
        if self.num_update % self.target_update_frequency == 0:
            for t_param, l_param in zip(
                    self.pi_target.parameters(), self.pi.parameters()
            ):
                t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        self.loss_calc_dict = {}    

    def act(self, state, obs, deterministic=False):
        with torch.no_grad():
            state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
            obs = torch.cat([obs, state_tile], dim=1)
            if deterministic:
                _, log_prob, a = self.pi.sample(obs)
            else:
                a, log_prob, _ = self.pi.sample(obs)
            a = a.to('cpu')
            return self.decodeActions(*[a[:, i] for i in range(self.n_a)])


