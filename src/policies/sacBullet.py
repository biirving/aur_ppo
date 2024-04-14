from src.policies.bulletArmPolicy import bulletArmPolicy
import torch
import torch.nn.functional as F
from copy import deepcopy
import numpy as np

class sacBullet(bulletArmPolicy):
    def __init__(self, alpha=1e-2, actor_lr=1e-3, critic_lr=1e-3, alpha_lr=1e-3, gamma=0.99, target_update_frequency=1):
        super().__init__()
        self.alpha = alpha
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.gamma = gamma
        self.target_update_frequency = target_update_frequency
        self.tau = 1e-2
        self.actor = None
        self.critic = None
        self.target_entropy = -self.n_a
        self.num_update = 0

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
    
    def compute_loss_q(self):
        batch_size, s, o, a, r, s2, o2, d, step_lefts, is_experts = self._loadLossCalcDict()
        with torch.no_grad():
            a_coded, logp_a2, mean = self.pi.sample(o2)
            logp_a2 = logp_a2.reshape(batch_size)
            q1_pi_targ, q2_pi_targ = self.critic_target(o2, a_coded)
            q1_pi_targ=q1_pi_targ.reshape(batch_size)
            q2_pi_targ=q2_pi_targ.reshape(batch_size)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ) - self.alpha * logp_a2
            backup = r + d * self.gamma * q_pi_targ
        q1, q2 = self.critic(o, a)
        q1 = q1.reshape(batch_size)
        q2 = q2.reshape(batch_size)
        loss_q1 = F.mse_loss(q1, backup)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        loss_q2 = F.mse_loss(q2, backup)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        return loss_q1, loss_q2
    
    def compute_loss_pi(self):
        batch_size, s, o, a, r, s2, o2, d, step_lefts, is_experts = self._loadLossCalcDict()
        a_coded, log_pi, mean = self.pi.sample(o)
        self.loss_calc_dict['pi'] = a_coded 
        self.loss_calc_dict['mean'] = mean
        self.loss_calc_dict['log_pi'] = log_pi
        q1_pi, q2_pi = self.critic(o, a_coded)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_pi = ((self.alpha * log_pi) - q_pi).mean()
        return loss_pi

    def update_critic(self):
        qf1_loss, qf2_loss  = self.compute_loss_q()
        qf_loss = qf1_loss + qf2_loss
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

    def update_actor(self):
        loss_pi = self.compute_loss_pi()
        log_pi = self.loss_calc_dict['log_pi']
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

    # Set up model saving
    def update(self, data):
        self._loadBatchToDevice(data)
        self.update_critic()
        self.update_actor()
        tau = self.tau
        self.num_update += 1
        if self.num_update % self.target_update_frequency == 0:
            for t_param, l_param in zip(
                    self.critic_target.parameters(), self.critic.parameters()
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
    
    def train(self):
        if self.pi is None or self.critic is None:
            raise ValueError('Agent not yet initialized.') 
        self.pi.train()
        self.critic.train()
