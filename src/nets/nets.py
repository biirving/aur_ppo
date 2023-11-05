import torch
from torch import nn, tensor
import numpy as np
from torch.distributions import Categorical, Normal, MultivariateNormal
import torch.nn.functional as F


# our actors, for the discrete and continuous spaces

class discrete_net(nn.Module):
    def __init__(self, dim:int, input_dim:int, output_dim:int, num_layers:int, dropout:float) -> None:
        super(discrete_net, self).__init__()
        layers = [nn.Linear(input_dim, dim), nn.Tanh(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(dim, dim), nn.Tanh(), nn.Dropout(dropout)])
        # use simple linear layer instead of softmax function
        layers.append(nn.Linear(dim, output_dim))
        layers.append(nn.Softmax(dim=0))
        self.net = nn.Sequential(*layers)
    def forward(self, input:tensor) -> tensor:
        return self.net(input)

class continuous_net(nn.Module):
    def __init__(self, dim:int, input_dim:int, output_dim:int, num_layers:int, dropout:float) -> None:
        super(discrete_net, self).__init__()
        layers = [nn.Linear(input_dim, dim), nn.Tanh(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(dim, dim), nn.Tanh(), nn.Dropout(dropout)])
        # use simple linear layer instead of softmax function
        layers.append(nn.Linear(dim, output_dim), nn.Tanh())
        self.net = nn.Sequential(*layers)
    def forward(self, input:tensor) -> tensor:
        return self.net(input)

class critic(nn.Module):
    def __init__(self, dim:int, input_dim:int, num_layers:int, dropout:float) -> None:
        super(Critic, self).__init__()
        layers = [nn.Linear(input_dim, dim), nn.Tanh(), nn.Dropout()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(dim, dim), nn.Tanh(), nn.Dropout(dropout)])
        # use simple linear layer instead of softmax function
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, input:tensor) -> tensor:
        return self.net(input)
