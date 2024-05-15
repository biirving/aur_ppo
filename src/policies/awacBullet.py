from src.policies.bulletArmPolicy import bulletArmPolicy
from src.policies.sacBullet import sacBullet
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import sys

class awacBullet(sacBullet):
    def __init__(self, alpha=1e-2, actor_lr=1e-3, critic_lr=1e-3, alpha_lr=1e-3, gamma=0.99, cql_scale=1e-4, target_update_frequency=1, awr_weight=1.0, consq=False):
        super().__init__(alpha, actor_lr, critic_lr, alpha_lr, gamma, target_update_frequency)
        self.cql_scale = cql_scale
        self.awr_weight = awr_weight
        self.consq = consq

    def initNet(self, actor, critic, encoder_type):
        self.pi=actor
        self.critic_1=critic
        self.critic_2=deepcopy(self.critic_1)
        self.encoder_type=encoder_type

        # TODO: Investigate alternative optimization options (grid search?)
        self.pi_optimizer = torch.optim.Adam(self.pi.parameters(), lr=self.actor_lr)
        self.q1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=self.critic_lr)
        self.q2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=self.critic_lr)

        self.critic_target_1 = deepcopy(self.critic_1)
        self.critic_target_2 = deepcopy(self.critic_2)

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
            q1_pi_targ = self.critic_target_1(o2, a_sampled)
            q2_pi_targ = self.critic_target_2(o2, a_sampled)
            q1_pi_targ=q1_pi_targ.reshape(batch_size)
            q2_pi_targ=q2_pi_targ.reshape(batch_size)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ) - self.alpha * logp_a2
            backup = r + (1. - d) * self.gamma * q_pi_targ
        
        q1 = self.critic_1(o, a)
        q2 = self.critic_2(o, a)
        q1 = q1.reshape(batch_size)
        q2 = q2.reshape(batch_size)

        loss_q1 = F.mse_loss(q1, backup.detach()) 
        loss_q2 = F.mse_loss(q2, backup.detach()) 

        # Conservative Q-Learning
        if self.consq:
            pass
            """
            q1_pi, q2_pi = self.critic(o2, a_sampled)
            log_sum_exp_q1 = torch.logsumexp(q1_pi, dim=0).mean()
            log_sum_exp_q2 = torch.logsumexp(q2_pi, dim=0).mean()
            conservative_loss_q1 = (log_sum_exp_q1 - q1.mean()) * self.cql_scale
            conservative_loss_q2 = (log_sum_exp_q2 - q2.mean()) * self.cql_scale
            loss_q1 += conservative_loss_q1 
            loss_q2 += conservative_loss_q2 
            """
        return loss_q1, loss_q2
    
    def update_critic(self):

        qf1_loss, qf2_loss  = self.compute_loss_q()

        self.q1_optimizer.zero_grad()
        qf1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        qf2_loss.backward()
        self.q2_optimizer.step()

       

    def compute_loss_pi(self):

        # scale of the weights
        beta = 2

        batch_size, s, o, a, r, s2, o2, d, step_lefts, is_experts, e_a = self._loadLossCalcDict()
        a_sampled, log_pi, mean = self.pi.sample(o)
        self.loss_calc_dict['pi'] = a_sampled 
        self.loss_calc_dict['mean'] = mean
        self.loss_calc_dict['log_pi'] = log_pi

        # match all of these up correctly
        q1 = self.critic_target_1(o, a)
        q2 = self.critic_target_2(o, a)

        # correct way to calculate q advantage
        old_q = torch.min(q1, q2)

        q1_pi = self.critic_1(o, a_sampled)
        q2_pi = self.critic_2(o, a_sampled)
        
        # Can also simply use q1_pi
        v_pi = torch.min(q1_pi, q2_pi)

        # the advantage
        adv = old_q - v_pi
        loss_pi = self.alpha * log_pi.mean()

        # so this is not working for some reason
        weights = F.softmax(adv / beta, dim=0)
        weights = weights[:, 0]

        # Why isn't the action keyword correctly processing
        _, policy_log_prob, _ = self.pi.sample(o, x_t=a)

        loss_pi = loss_pi + self.awr_weight * (-policy_log_prob * weights.shape[0] * weights.detach()).mean()
        return loss_pi

    def update(self, data):
        self._loadBatchToDevice(data)
        self.update_critic()
        self.update_actor()
        tau = self.tau
        self.num_update += 1
        if self.num_update % self.target_update_frequency == 0:
            for t_param, l_param in zip(
                    self.critic_target_1.parameters(), self.critic_1.parameters()
            ):
                t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)
            for t_param, l_param in zip(
                    self.critic_target_2.parameters(), self.critic_2.parameters()
            ):
                t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        self.loss_calc_dict = {} 
