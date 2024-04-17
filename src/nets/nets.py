import torch
from torch import nn, tensor
import numpy as np
from torch.distributions import Categorical, Normal, MultivariateNormal
import torch.nn.functional as F
import numpy as np


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# orthogonal initialization
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

class discrete_net(nn.Module):
    # at a torch datatype flag to allow changes to floating point size
    def __init__(self, dim:int, input_dim:int, output_dim:int, num_layers:int, dropout:float, action_std=0.01) -> None:
        super().__init__()
        layers = [layer_init(nn.Linear(np.array(input_dim).prod(), dim)), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([layer_init(nn.Linear(dim, dim)), nn.Tanh()])
        layers.append(layer_init(nn.Linear(dim, output_dim), action_std))
        self.net = nn.Sequential(*layers)
    def forward(self, input:tensor) -> tensor:
        return self.net(input)

class continuous_net(nn.Module):
    def __init__(self, dim:int, input_dim:int, output_dim:int, num_layers:int, dropout:float, action_std=0.01) -> None:
        super().__init__()
        layers = [layer_init(nn.Linear(np.array(input_dim).prod(), dim)), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([layer_init(nn.Linear(dim, dim)), nn.Tanh()])
        # use simple linear layer instead of softmax function
        layers.append(layer_init(nn.Linear(dim, np.prod(output_dim)), action_std))
        self.net = nn.Sequential(*layers)
    def forward(self, input:tensor) -> tensor:
        return self.net(input)

class critic(nn.Module):
    def __init__(self, dim:int, input_dim:int, num_layers:int, dropout:float, action_std=1.0) -> None:
        super().__init__()
        layers = [layer_init(nn.Linear(np.array(input_dim).prod(), dim)), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([layer_init(nn.Linear(dim, dim)), nn.Tanh()])
        # use simple linear layer instead of softmax function
        layers.append(layer_init(nn.Linear(dim, 1), action_std))
        self.net = nn.Sequential(*layers)
    def forward(self, input:tensor) -> tensor:
        return self.net(input)

class cnn(nn.Module):
    def __init__(self, dim:int, input_dim:int, num_layers:int, dropout:float, action_std=1.0) -> None:
        super().__init__()
        layers = [layer_init(nn.Linear(np.array(input_dim).prod(), dim)), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([layer_init(nn.Linear(dim, dim)), nn.Tanh()])
        # use simple linear layer instead of softmax function
        layers.append(layer_init(nn.Linear(dim, 1), action_std))
        self.net = nn.Sequential(*layers)
    def forward(self, input:tensor) -> tensor:
        return self.net(input)

class SACGaussianPolicyBase(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def sample(self, x):
        mean, log_std = self.forward(x)
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


class PPOGaussianPolicyBase(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def sample(self, x, action=None):
        mean, log_std = self.forward(x)
        action_std = log_std.exp()
        dist = Normal(mean, action_std)
        if action is None:
            action = dist.rsample()
        action = torch.tanh(action)
        log_prob = dist.log_prob(action)
        # clipping the log probability
        log_prob -= torch.log((1 - action.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        entropy = dist.entropy()		
        return action, log_prob, mean, entropy 