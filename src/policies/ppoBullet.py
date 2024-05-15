from src.policies.bulletArmPolicy import bulletArmPolicy
import torch
from torch import nn
import numpy as np
from copy import deepcopy
import sys

class ppoBullet(bulletArmPolicy):
    def __init__(self, alpha=1e-2, actor_lr=1e-3, critic_lr=1e-3, alpha_lr=1e-3, 
    gamma=0.99, gae=True, target_update_frequency=1, num_processes=5, total_steps=10000, 
    update_epochs=10, clip_coeff=0.2, max_grad_norm=0.5, value_coeff=0.5, expert_weight=0.01,
    entropy_coeff=0.01, gae_lambda=0.95, clip_vloss=False, norm_adv=True, num_minibatches=32,
    target_kl=0.01):
        super().__init__()
        self.alpha = alpha
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.target_update_frequency = target_update_frequency
        self.tau = 1e-2

        # TODO: Add these to abstract PPO class
        self.gae = gae
        self.gamma = gamma
        self.minibatch_size = int(num_processes * total_steps) // num_minibatches
        self.num_update_epochs = update_epochs
        self.clip_coeff = clip_coeff
        self.max_grad_norm = max_grad_norm
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.clip_vloss = clip_vloss
        self.norm_adv = norm_adv
        self.gae_lambda = gae_lambda
        self.num_processes = num_processes
        self.expert_weight = expert_weight
        self.flattened_buffer = {}
        self.transition_dict = {}
        self.target_kl = target_kl

    def initNet(self, actor, critic, encoder_type):
        self.pi=actor
        self.critic=critic
        self.encoder_type=encoder_type

        # TODO: Investigate alternative optimization options (grid search?)
        self.pi_optimizer =  torch.optim.Adam([
                        {'params': self.pi.parameters(), 'lr': self.actor_lr}
                        #{'params': self.critic.parameters(), 'lr': self.critic_lr}
                    ])
        # should use two separate optimizers
        self.v_optimizer =  torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        #self.pi_target = deepcopy(self.pi)

    def _loadBatchToDevice(self, batch, device='cuda'):
        states_tensor = torch.tensor(np.stack([d.state for d in batch])).long().to(device)
        obs_tensor = torch.tensor(np.stack([d.obs for d in batch])).to(device)
        if len(obs_tensor.shape) == 3:
            obs_tensor = obs_tensor.unsqueeze(1)
        action_tensor = torch.tensor(np.stack([d.action for d in batch])).to(device)
        rewards_tensor = torch.tensor(np.stack([d.reward.squeeze() for d in batch])).to(device)
        dones_tensor = torch.tensor(np.stack([d.done for d in batch])).int()
        non_final_masks = (dones_tensor ^ 1).float().to(device)
        step_lefts_tensor = torch.tensor(np.stack([d.step_left for d in batch])).to(device)
        values_tensor = torch.tensor(np.stack([d.value for d in batch])).to(device)
        expert_actions_tensor = torch.tensor(np.stack([d.expert_action for d in batch])).to(device)
        log_probs_tensor = torch.tensor(np.stack([d.log_probs for d in batch])).to(device)

        # scale observation from int to float
        obs_tensor = obs_tensor/255*0.4

        self.loss_calc_dict['batch_size'] = int(len(batch) / self.num_processes)
        self.loss_calc_dict['states'] = states_tensor
        self.loss_calc_dict['obs'] = obs_tensor
        self.loss_calc_dict['actions'] = action_tensor
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
        actions = self.loss_calc_dict['actions']
        rewards = self.loss_calc_dict['rewards']
        non_final_masks = self.loss_calc_dict['non_final_masks']
        step_lefts = self.loss_calc_dict['step_lefts']
        values = self.loss_calc_dict['values']
        expert_actions = self.loss_calc_dict['expert_actions']
        log_probs = self.loss_calc_dict['log_probs']
        return batch_size, states, obs, actions, rewards, non_final_masks, step_lefts, values, expert_actions, log_probs

    def _loadLossCalcDict(self):
        """
        get the loaded batch data in self.loss_calc_dict
        :return: batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts
        """
        batch_size, states, obs, actions, rewards, non_final_masks, step_lefts, values, expert_actions, log_probs= self.load_info() 

        if self.obs_type == 'pixel':
            # stack state as the second channel of the obs
            obs = torch.cat([obs, states.reshape(states.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])], dim=1)
        return batch_size, states, obs, actions, rewards, non_final_masks, step_lefts, values, expert_actions, log_probs

    def get_buffer_values(self, inds, device='cuda'):
        states = self.loss_calc_dict['states'][inds].to(device)
        obs = self.loss_calc_dict['obs'][inds].to(device)
        obs = torch.cat([obs, states.reshape(states.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])], dim=1)
        old_log_probs = self.loss_calc_dict['log_probs'][inds].to(device)
        actions = self.loss_calc_dict['actions'][inds].to(device)
        expert = self.loss_calc_dict['expert_actions'][inds].to(device)
        advantages = self.loss_calc_dict['advantages'][inds].to(device)
        returns = self.loss_calc_dict['returns'][inds].to(device)
        values = self.loss_calc_dict['values'][inds].to(device)
        return states, obs, old_log_probs, actions, expert, advantages, returns, values

    def run_gae(self, next_value, next_done):
        lastgaelam = 0

        batch_size, states, obs, actions, rewards, dones, steps_left, values, expert, log_probs = self._loadLossCalcDict()

        # Reshaping due to nature of buffer
        advantages = torch.zeros(batch_size, self.num_processes).cuda()
        rewards = rewards.reshape(batch_size, self.num_processes).cuda()
        dones = dones.reshape(batch_size, self.num_processes).cuda()
        values = values.reshape(batch_size, self.num_processes).cuda()
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
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
        what = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values

        # flatten these before returning
        returns = returns.reshape(-1)
        advantages = advantages.reshape(-1)
        return returns, advantages

    # these can be stored in a separate file
    def normal_advantage(self, next_value, next_done):
        batch_size, states, obs, actions, rewards, dones, steps_left, values, expert, log_probs = self._loadLossCalcDict()
        # the default rollout scheme
        returns = torch.zeros_like(rewards).to(device)
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
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
            next_value = self.critic(next_obs).flatten()
            if self.gae:
                returns, advantages = self.run_gae(next_value, next_done)
            else:
                returns, advantages = self.normal_advantage(next_value, next_done)
        return returns, advantages
    
    def compute_loss_pi(self, mb_inds):

        states, obs, old_log_probs, actions, expert, advantages, returns, values = self.get_buffer_values(mb_inds)
        # we want our gradients collected directly
        a, newlogprob, mean, entropy = self.pi.sample(obs.cuda())
        old_log_probs = old_log_probs.squeeze(dim=1)
        newlogprob = newlogprob.squeeze(dim=1)
        print('new log prob', newlogprob)
        log_ratio = newlogprob - old_log_probs
        print('log ratio', log_ratio)
        ratio = log_ratio.exp()
        mb_advantages = advantages[mb_inds]
        # 1e-8 avoids division by 0
        print('prenorm adv', mb_advantages)

        if self.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean())/(mb_advantages.std() + 1e-8)

        print('post norm adv', mb_advantages)
        loss_one = -mb_advantages * ratio

        print('ratio', ratio)
        loss_two = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coeff, 1 + self.clip_coeff)
        policy_loss = torch.max(loss_one, loss_two).mean()

        print('policy loss', policy_loss)
        # want to make sure that this expert loss is being computed correctly
        expert_loss = nn.functional.mse_loss(a, expert)
        # adding the mse here
        loss = policy_loss + self.expert_weight * expert_loss 

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-log_ratio).mean()
            approx_kl = ((ratio - 1) - log_ratio).mean()
        return loss, entropy, approx_kl

    def compute_loss_v(self, mb_inds):
        states, obs, old_log_probs, actions, expert, advantages, returns, values = self.get_buffer_values(mb_inds)
        # the only difference between the buffer action and the non buffer action is that the buffer
        # action has been passed through decode actions (which can affect the gradient?)
        a, newlogprob, mean, entropy = self.pi.sample(obs.cuda())
        newvalue = self.critic(obs.cuda())
        if self.clip_vloss:
            v_loss_unclipped = (newvalue - returns) ** 2
            v_clipped = values + torch.clamp(
                newvalue - values,
                -self.clip_coeff,
                self.clip_coeff,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            value_loss = 0.5 * v_loss_max.mean()
        else:
            value_loss = 0.5 * ((newvalue - returns) ** 2).mean()
        value_loss *= self.value_coeff
        return value_loss

    # TODO: Most of this method should be in policy runner
    # Implement gradient accumulation
    def update(self, data, next_obs, next_done, dists=None):
        self._loadBatchToDevice(data)
        if dists is not None:
            # Add distance to partial reward- ppo should function better
            print(self.loss_calc_dict['rewards'])
            # is this the correct way to incorporate the distances?
            self.loss_calc_dict['rewards'] += 1 - dists

        batch_size, states, obs, actions, rewards, dones, steps_left, values, expert, log_probs = self._loadLossCalcDict()

        # this calculation is computationally expensive
        next_value = self.critic(obs)
        returns, advantages = self.advantages(next_obs, next_value, next_done)

        self.loss_calc_dict['returns'] = returns
        self.loss_calc_dict['advantages'] = advantages

        for uep in range(self.num_update_epochs):
            b_inds = np.arange(batch_size)
            for index in range(0, batch_size, self.minibatch_size): 
                mb_inds = b_inds[index:index+self.minibatch_size]

                loss, entropy, approx_kl = self.compute_loss_pi(mb_inds)
                entropy_loss = entropy.mean()
                # how to track the gradients?
                print(loss)
                print(entropy_loss)
                pi_loss = loss - self.entropy_coeff * entropy_loss 
                print('pi loss', pi_loss)
                self.pi_optimizer.zero_grad()
                pi_loss.backward()
                torch.nn.utils.clip_grad_value_(self.pi.parameters(), clip_value=1.0) 
                self.pi_optimizer.step()

                for name, param in self.pi.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f"{name} gradient: \n {param.grad}")
                            print(f"Warning: NaN detected in the gradients of {name}")
                            sys.exit()

                v_loss = self.compute_loss_v(mb_inds)
                self.v_optimizer.zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_value_(self.critic.parameters(), clip_value=1.0) 
                self.v_optimizer.step()

                for name, param in self.critic.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f"{name} gradient: \n {param.grad}")
                            print(f"Warning: NaN detected in the gradients of {name}")
                            sys.exit()

            if approx_kl > self.target_kl:
                break

            #self.pi_target.load_state_dict(self.pi.state_dict())
        self.loss_calc_dict = {}    

    def act(self, states, obs, deterministic=False):
        with torch.no_grad():
            obs = torch.cat([obs, states.reshape(states.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])], dim=1)
            mean=None
            if deterministic:
                _, log_prob, a, entropy = self.pi.sample(obs)
            else:
                a, log_prob, mean, entropy = self.pi.sample(obs)
            a = a.to('cpu')
            if mean is not None:
                mean=mean.cpu()
            val = self.critic(obs)
            return self.decodeActions(*[a[:, i] for i in range(self.n_a)]), log_prob.cpu(), mean, val.cpu()

    def pretrain_update(self, obs, expert):
        """
        parameters: loss 

        We only update the policy during pretraining. 
        """
        a, log_prob, mean, entropy = self.pi.sample(obs)
        expert_loss = nn.functional.mse_loss(a, expert)
        self.pi_optimizer.zero_grad()
        expert_loss.backward()
        nn.utils.clip_grad_norm_(self.pi.parameters(), self.max_grad_norm)
        self.pi_optimizer.step()
        for name, param in self.pi.named_parameters():
            # check if the gradient is not none
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    #print(f"{name} gradient: \n {param.grad}")
                    print(f"Warning: NaN detected in the gradients of {name}")
                    sys.exit()

    def save_agent(self, path):
        torch.save(self.pi.state_dict(), path + '/' + env + '_' + self.encoder_type +  '_agent.pt')
        torch.save(self.critic.state_dict(), path + '/' + env + '_' + self.encoder_type + '_critic.pt')
