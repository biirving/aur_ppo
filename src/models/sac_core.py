import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


from nets.equiv import EquivariantActor, EquivariantCritic
from nets.base_cnns import base_actor, base_critic, base_encoder


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

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        #self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.net = base_actor()
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    # changes to support robot
    # courtesy of Dian Wang
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

	# scaled actions
    def getActionFromPlan(self, plan):
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
        

    def forward(self, state, obs, deterministic=False, with_logprob=True):
        state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
        cat_obs = torch.cat([obs, state_tile], dim=1).to(self.device)
        net_out = self.net(cat_obs)
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
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


# is this the critic
# this is a value function
class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)
        
    # so the q function is processing the action, observation, and state
    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


# we need to add support for our CNNs
class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    # add the observation in here
    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()