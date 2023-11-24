

# advantages
def run_gae(self, next_value, next_done):
    advantages = torch.zeros_like(self.buffer.rewards).to(device)
    lastgaelam = 0
    for t in reversed(range(self.num_steps)):
        if t == self.num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - self.buffer.terminals[t + 1]
            nextvalues = self.buffer.values[t + 1]
        # If episode finishes after timestep t, we mask the value and the previous advantage value, 
        # so that advantages[t] stores the (reward - value) at that timestep without
        # taking a value from the next episode, or storing a value in this episode to be multiplied
        # in the calculation of the rollout of the next episode
        delta = self.buffer.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.buffer.values[t]
        advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + self.buffer.values
    return returns, advantages

# these can be stored in a separate file
def normal_advantage(self, next_value, next_done):
    # the default rollout scheme
    returns = torch.zeros_like(self.buffer.rewards).to(device)
    for t in reversed(range(self.num_steps)):
        if t == self.num_steps - 1:
            nextnonterminal = 1.0 - next_done
            next_return = next_value
        else:
            nextnonterminal = 1.0 - self.buffer.terminals[t + 1]
            next_return = returns[t + 1]
        returns[t] = self.buffer.rewards[t] + self.gamma * nextnonterminal * next_return
    advantages = returns - self.buffer.values
    return returns, advantages

def advantages(self, next_obs, next_done):
    with torch.no_grad():
        next_value = self.policy_old.critic(next_obs).reshape(1, -1)
        if self.gae:
            returns, advantages = self.run_gae(next_value, next_done)
        else:
            returns, advantages = self.run_normal_advantage(next_value, next_done)
    return returns, advantages

