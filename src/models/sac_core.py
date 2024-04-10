import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


from nets.equiv import EquivariantActor, EquivariantCritic, EquivariantSACCritic
from nets.base_cnns import base_actor, base_critic, base_encoder


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
class MLPActorCritic(nn.Module):
    def __init__(self, n_a=5, hidden_sizes=(1024,1024), activation=nn.ReLU):
        super().__init__()

        # build policy and value functions
        #self.pi = SquashedGaussianMLPActor(hidden_sizes, activation)
        self.pi = EquivariantActor(N=8)
        self.critic = EquivariantSACCritic(N=8)
        #self.q2 = EquivariantSACCritic()
    
    def sample(self, x):
        mean, log_std = self.pi.forward(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean

    def act(self, state, obs, deterministic=False):
        with torch.no_grad():
            state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
            cat_obs = torch.cat([obs, state_tile], dim=1)
            a, log_prob, mean = self.sample(cat_obs)
            return a.cpu()
