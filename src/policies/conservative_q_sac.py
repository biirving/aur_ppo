from src.policies.awacBullet import awacBullet
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import sys

class conservativeQSACBullet(awacBullet):
    def __init__(self, alpha=1e-2, actor_lr=1e-3, critic_lr=1e-3, alpha_lr=1e-3, gamma=0.99, cql_scale=1e-4, target_update_frequency=1, awr_weight=1.0, consq=False):
        super().__init__(alpha, actor_lr, critic_lr, alpha_lr, gamma, target_update_frequency)
        self.cql_scale = cql_scale
        self.awr_weight = awr_weight
        self.consq = consq

    def compute_loss_q(self):
        batch_size, s, o, a, r, s2, o2, d, step_lefts, is_experts, e_a = self._loadLossCalcDict()
        with torch.no_grad():
            a_sampled, logp_a2, mean = self.pi.sample(o2)
            logp_a2 = logp_a2.reshape(batch_size)
            q1_pi_targ, q2_pi_targ = self.critic_target(o2, a_sampled)
            q1_pi_targ=q1_pi_targ.reshape(batch_size)
            q2_pi_targ=q2_pi_targ.reshape(batch_size)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ) - self.alpha * logp_a2
            backup = r + (1. - d) * self.gamma * q_pi_targ
        q1, q2 = self.critic(o, a)
        q1 = q1.reshape(batch_size)
        q2 = q2.reshape(batch_size)

        loss_q1 = F.mse_loss(q1, backup.detach()) 
        loss_q2 = F.mse_loss(q2, backup.detach()) 

        cql_random_actions = a.new_empty((batch_size, ))

        # Conservative Q-Learning
        if self.consq:
            q1_pi, q2_pi = self.critic(o2, a_sampled)
            log_sum_exp_q1 = torch.logsumexp(q1_pi, dim=0).mean()
            log_sum_exp_q2 = torch.logsumexp(q2_pi, dim=0).mean()
            conservative_loss_q1 = (log_sum_exp_q1 - q1.mean()) * self.cql_scale
            conservative_loss_q2 = (log_sum_exp_q2 - q2.mean()) * self.cql_scale
            loss_q1 += conservative_loss_q1 
            loss_q2 += conservative_loss_q2 
        
        return loss_q1, loss_q2
    
