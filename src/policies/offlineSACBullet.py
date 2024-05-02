from src.policies.bulletArmPolicy import bulletArmPolicy
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import sys

class offlineSACBullet(bulletArmPolicy):
    def __init__(self, alpha=1e-2, actor_lr=1e-3, critic_lr=1e-3, alpha_lr=1e-3, gamma=0.99, cql_scale=1e-4, target_update_frequency=1):
        super().__init__()
        self.alpha = alpha
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.gamma = gamma
        self.target_update_frequency = target_update_frequency
        self.tau = 1e-2
        self.pi = None
        self.critic = None
        self.target_entropy = -self.n_a
        self.num_update = 0
        self.cql_scale = cql_scale

    def initNet(self, actor, critic):
        self.pi=actor
        self.critic=critic

        # TODO: Investigate alternative optimization options (grid search?)
        self.pi_optimizer = torch.optim.Adam(self.pi.parameters(), lr=self.actor_lr)
        self.q_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.critic_target = deepcopy(self.critic)

        # entropy tuning. see https://arxiv.org/abs/1801.01290
        self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=torch.device('cuda'))
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)

    def _loadBatchToDevice(self, batch, device='cuda'):

        states_tensor = torch.tensor(np.stack([d.state for d in batch])).long().to(device)
        obs_tensor = torch.tensor(np.stack([d.obs for d in batch])).to(device)
        if len(obs_tensor.shape) == 3:
            obs_tensor = obs_tensor.unsqueeze(1)
        action_tensor = torch.tensor(np.stack([d.action for d in batch])).to(device)
        rewards_tensor = torch.tensor(np.stack([d.reward.squeeze() for d in batch])).to(device)
        next_states_tensor = torch.tensor(np.stack([d.next_state for d in batch])).long().to(device)
        next_obs_tensor = torch.tensor(np.stack([d.next_obs for d in batch])).to(device)
        if len(next_obs_tensor.shape) == 3:
            next_obs_tensor = next_obs_tensor.unsqueeze(1)
        dones_tensor = torch.tensor(np.stack([d.done for d in batch])).int()
        non_final_masks = (dones_tensor ^ 1).float().to(device)
        step_lefts_tensor = torch.tensor(np.stack([d.step_left for d in batch])).to(device)
        is_experts_tensor = torch.tensor(np.stack([d.expert for d in batch])).bool().to(device)
        expert_actions = torch.tensor(np.stack([d.expert_action for d in batch])).bool().to(device)

        # scale observation from int to float
        obs_tensor = obs_tensor/255*0.4
        next_obs_tensor = next_obs_tensor/255*0.4

        self.loss_calc_dict['batch_size'] = len(batch)
        self.loss_calc_dict['states'] = states_tensor
        self.loss_calc_dict['obs'] = obs_tensor
        self.loss_calc_dict['action_idx'] = action_tensor
        self.loss_calc_dict['rewards'] = rewards_tensor
        self.loss_calc_dict['next_states'] = next_states_tensor
        self.loss_calc_dict['next_obs'] = next_obs_tensor
        self.loss_calc_dict['non_final_masks'] = non_final_masks
        self.loss_calc_dict['step_lefts'] = step_lefts_tensor
        self.loss_calc_dict['is_experts'] = is_experts_tensor
        self.loss_calc_dict['expert_actions'] = expert_actions
        
        return states_tensor, obs_tensor, action_tensor, rewards_tensor, next_states_tensor, \
               next_obs_tensor, non_final_masks, step_lefts_tensor, is_experts_tensor, expert_actions

    def load_info(self):
        """
        Get the loaded batch data in self.loss_calc_dict
        :return: batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts
        """
        batch_size = self.loss_calc_dict['batch_size']
        states = self.loss_calc_dict['states']
        obs = self.loss_calc_dict['obs']
        action_idx = self.loss_calc_dict['action_idx']
        rewards = self.loss_calc_dict['rewards']
        next_states = self.loss_calc_dict['next_states']
        next_obs = self.loss_calc_dict['next_obs']
        non_final_masks = self.loss_calc_dict['non_final_masks']
        step_lefts = self.loss_calc_dict['step_lefts']
        is_experts = self.loss_calc_dict['is_experts']
        expert_actions = self.loss_calc_dict['expert_actions']
        return batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts, expert_actions

    def _loadLossCalcDict(self):
        """
        get the loaded batch data in self.loss_calc_dict
        :return: batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts
        """
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts, expert_actions = self.load_info() 

        if self.obs_type == 'pixel':
            # stack state as the second channel of the obs
            obs = torch.cat([obs, states.reshape(states.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])], dim=1)
            next_obs = torch.cat([next_obs, next_states.reshape(next_states.size(0), 1, 1, 1).repeat(1, 1, next_obs.shape[2], next_obs.shape[3])], dim=1)

        return batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts, expert_actions

    def compute_loss_q(self):
        batch_size, s, o, a, r, s2, o2, d, step_lefts, is_experts, e_a = self._loadLossCalcDict()
        with torch.no_grad():
            a_sampled, logp_a2, mean = self.pi.sample(o2)
            logp_a2 = logp_a2.reshape(batch_size)
            q1_pi_targ, q2_pi_targ = self.critic_target(o2, a_sampled)
            q1_pi_targ=q1_pi_targ.reshape(batch_size)
            q2_pi_targ=q2_pi_targ.reshape(batch_size)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ) - self.alpha * logp_a2
            backup = r + (1 - d) * self.gamma * q_pi_targ
        q1, q2 = self.critic(o, a)
        q1 = q1.reshape(batch_size)
        q2 = q2.reshape(batch_size)

        # Conservative Q-Learning
        q1_pi, q2_pi = self.critic(o2, a_sampled)
        log_sum_exp_q1 = torch.logsumexp(q1_pi, dim=0).mean()
        log_sum_exp_q2 = torch.logsumexp(q2_pi, dim=0).mean()
        conservative_loss_q1 = (log_sum_exp_q1 - q1.mean()) * self.cql_scale
        conservative_loss_q2 = (log_sum_exp_q2 - q2.mean()) * self.cql_scale

        loss_q1 = F.mse_loss(q1, backup) + log_sum_exp_q1 
        loss_q2 = F.mse_loss(q2, backup) + log_sum_exp_q2 

        return loss_q1, loss_q2
    
    def compute_loss_pi(self):
        batch_size, s, o, a, r, s2, o2, d, step_lefts, is_experts, e_a = self._loadLossCalcDict()
        a_sampled, log_pi, mean = self.pi.sample(o)
        self.loss_calc_dict['pi'] = a_sampled 
        self.loss_calc_dict['mean'] = mean
        self.loss_calc_dict['log_pi'] = log_pi
        q1_pi, q2_pi = self.critic(o, a_sampled)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_pi = ((self.alpha * log_pi) - q_pi).mean()
        return loss_pi

