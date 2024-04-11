import numpy as np

from copy import deepcopy
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


from nets.equiv import EquivariantActor, EquivariantCritic, EquivariantSACCritic, EquivariantSACActor
from nets.base_cnns import base_actor, base_critic, base_encoder

print(torch.__version__)

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, hidden_sizes, activation, 
        dx=0.02, 
        dy=0.02, 
        dz=0.02, 
        dr=np.pi/8, 
        n_a=5, 
        tau=0.001,):
        super().__init__()

        self.p_range = torch.tensor([0, 1])
        self.dtheta_range = torch.tensor([-dr, dr])
        self.dx_range = torch.tensor([-dx, dx])
        self.dy_range = torch.tensor([-dy, dy])
        self.dz_range = torch.tensor([-dz, dz])	
        self.n_a = n_a
        #self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        
        #
        self.net = base_encoder(out_dim=128)

        # hidden_sizes
        self.mu_layer = nn.Linear(128, 5)
        self.log_std_layer = nn.Linear(128, 5)
        #self.act_limit = act_limit



    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)

        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:

            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        #pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    # how does this q function actually work?
    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


class SACCritic(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5):
        super().__init__()
        self.state_conv_1 = base_encoder(obs_shape=(2, 128, 128), out_dim=128)
        self.critic_fc_1 = torch.nn.Sequential(
            torch.nn.Linear(128 + action_dim, 128),
            nn.ReLU(inplace=True),
            torch.nn.Linear(128, 1)
        )
        self.apply(weights_init)

    def forward(self, obs, act):
        conv_out = self.state_conv_1(obs)
        out_1 = self.critic_fc_1(torch.cat((conv_out, act), dim=1))
        return out_1 

device = torch.device('cuda')
class MLPActorCritic:
    def __init__(self, n_a=5, hidden_sizes=(1024,1024), activation=nn.ReLU):
        super().__init__()

        # build policy and value functions
        #self.pi = SquashedGaussianMLPActor(hidden_sizes, activation)
        self.pi = EquivariantSACActor(N=8).cuda()
        self.critic = EquivariantSACCritic(N=8).cuda()
        #self.q2 = EquivariantSACCritic()
        dpos = 0.05
        dr=np.pi/8
        self.p_range = torch.tensor([0, 1])
        self.dtheta_range = torch.tensor([-dr, dr])
        self.dx_range = torch.tensor([-dpos, dpos])
        self.dy_range = torch.tensor([-dpos, dpos])
        self.dz_range = torch.tensor([-dpos, dpos])	
        self.n_a = n_a
        lr = 1e-3
        self.alpha = 1e-2
        self.gamma = 0.99
        self.num_update = 0
        self.pi_optimizer =  torch.optim.Adam(self.pi.parameters(), lr=lr)
        self.q_optimizer =  torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.critic_target = deepcopy(self.critic.to(device))
        self.batch_size=64

    def _loadBatchToDevice(self, batch, device=torch.device('cuda')):
        """
        Load batch into pytorch tensor
        :param batch: list of transitions
        :return: states_tensor, obs_tensor, action_tensor, rewards_tensor, next_states_tensor, next_obs_tensor,
                 non_final_masks, step_lefts_tensor, is_experts_tensor
        """
        states = []
        images = []
        xys = []
        rewards = []
        next_states = []
        next_obs = []
        dones = []
        step_lefts = []
        is_experts = []
        for d in batch:
            states.append(d.state)
            images.append(d.obs)
            xys.append(d.action)
            rewards.append(d.reward.squeeze())
            next_states.append(d.next_state)
            next_obs.append(d.next_obs)
            dones.append(d.done)
            step_lefts.append(d.step_left)
            is_experts.append(d.expert)
        states_tensor = torch.tensor(np.stack(states)).long().to(device)
        obs_tensor = torch.tensor(np.stack(images)).to(device)
        if len(obs_tensor.shape) == 3:
            obs_tensor = obs_tensor.unsqueeze(1)
        action_tensor = torch.tensor(np.stack(xys)).to(device)
        rewards_tensor = torch.tensor(np.stack(rewards)).to(device)
        next_states_tensor = torch.tensor(np.stack(next_states)).long().to(device)
        next_obs_tensor = torch.tensor(np.stack(next_obs)).to(device)
        if len(next_obs_tensor.shape) == 3:
            next_obs_tensor = next_obs_tensor.unsqueeze(1)
        dones_tensor = torch.tensor(np.stack(dones)).int()
        non_final_masks = (dones_tensor ^ 1).float().to(device)
        step_lefts_tensor = torch.tensor(np.stack(step_lefts)).to(device)
        is_experts_tensor = torch.tensor(np.stack(is_experts)).bool().to(device)

        # scale observation from int to float
        obs_tensor = obs_tensor/255*0.4
        next_obs_tensor = next_obs_tensor/255*0.4
        return states_tensor, obs_tensor, action_tensor, rewards_tensor, next_states_tensor, \
               next_obs_tensor, non_final_masks, step_lefts_tensor, is_experts_tensor

    def train(self):
        self.pi.train()
        self.critic.train()

    def decodeActions(self, *args):
        unscaled_p = args[0]
        unscaled_dx = args[1]
        unscaled_dy = args[2]
        unscaled_dz = args[3]

        p = 0.5 * (unscaled_p + 1) * (self.p_range[1] - self.p_range[0]) + self.p_range[0]
        dx = 0.5 * (unscaled_dx + 1) * (self.dx_range[1] - self.dx_range[0]) + self.dx_range[0]
        dy = 0.5 * (unscaled_dy + 1) * (self.dy_range[1] - self.dy_range[0]) + self.dy_range[0]
        dz = 0.5 * (unscaled_dz + 1) * (self.dz_range[1] - self.dz_range[0]) + self.dz_range[0]

        if self.n_a == 5:
            unscaled_dtheta = args[4]
            dtheta = 0.5 * (unscaled_dtheta + 1) * (self.dtheta_range[1] - self.dtheta_range[0]) + self.dtheta_range[0]
            actions = torch.stack([p, dx, dy, dz, dtheta], dim=1)
            unscaled_actions = torch.stack([unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz, unscaled_dtheta], dim=1)
        else:
            actions = torch.stack([p, dx, dy, dz], dim=1)
            unscaled_actions = torch.stack([unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz], dim=1)
        return unscaled_actions, actions

    def getActionFromPlan(self, plan):
        """
        Get unscaled and scaled actions from scaled planner action
        :param plan: scaled planner action (in true scale)
        :return: unscaled_actions (in range (-1, 1)), actions (in true scale)
        """
        def getUnscaledAction(action, action_range):
            unscaled_action = 2 * (action - action_range[0]) / (action_range[1] - action_range[0]) - 1
            return unscaled_action
        dx = plan[:, 1].clamp(*self.dx_range)
        p = plan[:, 0].clamp(*self.p_range)
        dy = plan[:, 2].clamp(*self.dy_range)
        dz = plan[:, 3].clamp(*self.dz_range)

        unscaled_p = getUnscaledAction(p, self.p_range)
        unscaled_dx = getUnscaledAction(dx, self.dx_range)
        unscaled_dy = getUnscaledAction(dy, self.dy_range)
        unscaled_dz = getUnscaledAction(dz, self.dz_range)

        if self.n_a == 5:
            dtheta = plan[:, 4].clamp(*self.dtheta_range)
            unscaled_dtheta = getUnscaledAction(dtheta, self.dtheta_range)
            return self.decodeActions(unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz, unscaled_dtheta)
        else:
            return self.decodeActions(unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz)

    def compute_loss_q(self, s, o, a, r, s2, o2, d):
        s_tile = s.reshape(s.size(0), 1, 1, 1).repeat(1, 1, o.shape[2], o.shape[3])
        cat_o = torch.cat([o, s_tile], dim=1).to(device)
        s_tile_2 = s2.reshape(s2.size(0), 1, 1, 1).repeat(1, 1, o2.shape[2], o2.shape[3])
        cat_o_2 = torch.cat([o2, s_tile_2], dim=1).to(device)
        with torch.no_grad():
            a_coded, logp_a2, mean = self.pi.sample(cat_o_2)
            logp_a2 = logp_a2.reshape(self.batch_size)
            q1_pi_targ, q2_pi_targ = self.critic_target(cat_o_2, a_coded)
            q1_pi_targ=q1_pi_targ.reshape(self.batch_size)
            q2_pi_targ=q2_pi_targ.reshape(self.batch_size)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ).to(device)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)
        q1, q2 = self.critic(cat_o, a)
        q1 = q1.reshape(self.batch_size)
        q2 = q2.reshape(self.batch_size)
        loss_q1 = F.mse_loss(q1, backup)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        loss_q2 = F.mse_loss(q2, backup)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        loss_q = loss_q1 + loss_q2

        q_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                      Q2Vals=q2.cpu().detach().numpy())

        return loss_q, q_info

    def compute_loss_pi(self, s, o):
        #with torch.no_grad():
        s_tile = s.reshape(s.size(0), 1, 1, 1).repeat(1, 1, o.shape[2], o.shape[3])
        cat_o = torch.cat([o, s_tile], dim=1).to(device)
        a_coded, logp_pi, mean = self.pi.sample(cat_o)
        q1_pi, q2_pi = self.critic(cat_o, a_coded)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_pi = (self.alpha * logp_pi - q_pi).mean()
        pi_info = dict(LogPi=logp_pi.cpu().detach().numpy())
        return loss_pi, pi_info

    # Set up model saving
    def update(self, data):
        s, o, a, r, s2, o2, d, _, _ = self._loadBatchToDevice(data)
        loss_q, q_info = self.compute_loss_q(s, o, a, r, s2, o2, d)
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()
        loss_pi, pi_info = self.compute_loss_pi(s, o)
        # is this correct
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()
        tau = 0.001
        self.num_update += 1
        if self.num_update % 100 == 0:
            for t_param, l_param in zip(
                    self.critic_target.parameters(), self.critic.parameters()
            ):
                t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)


    def act(self, state, obs, deterministic=False):
        with torch.no_grad():
            state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
            cat_obs = torch.cat([obs, state_tile], dim=1)
            a, log_prob, mean = self.pi.sample(cat_obs)
            a = a.cpu()
            return self.decodeActions(*[a[:, i] for i in range(self.n_a)])
