import torch
from torch import nn, tensor
import numpy as np
from torch.distributions import Categorical, Normal, MultivariateNormal
import torch.nn.functional as F
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

class discrete_net(nn.Module):
    def __init__(self, dim:int, input_dim:int, output_dim:int, num_layers:int, dropout:float) -> None:
        super().__init__()
        layers = [layer_init(nn.Linear(input_dim, dim)), nn.Tanh(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers.extend([layer_init(nn.Linear(dim, dim)), nn.Tanh(), nn.Dropout(dropout)])
        layers.append(layer_init(nn.Linear(dim, output_dim)))
        layers.append(nn.Softmax(dim=0))
        self.net = nn.Sequential(*layers)
    
    def forward(self, input:tensor) -> tensor:
        return self.net(input / 255.0)


class continuous_net(nn.Module):
    def __init__(self, dim:int, input_dim:int, output_dim:int, num_layers:int, dropout:float) -> None:
        super().__init__()
        layers = [layer_init(nn.Linear(input_dim, dim)), nn.Tanh(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers.extend([layer_init(nn.Linear(dim, dim)), nn.Tanh(), nn.Dropout(dropout)])
        # use simple linear layer instead of softmax function
        layers.append(layer_init(nn.Linear(dim, np.prod(output_dim))), std=0.01)
        self.net = nn.Sequential(*layers)

    def forward(self, input:tensor) -> tensor:
        return self.net(input)

class critic(nn.Module):
    def __init__(self, dim:int, input_dim:int, num_layers:int, dropout:float) -> None:
        super().__init__()
        layers = [layer_init(nn.Linear(input_dim, dim)), nn.Tanh(), nn.Dropout()]
        for _ in range(num_layers - 1):
            layers.extend([layer_init(nn.Linear(dim, dim)), nn.Tanh(), nn.Dropout(dropout)])
        # use simple linear layer instead of softmax function
        layers.append(layer_init(nn.Linear(dim, 1)))
        self.net = nn.Sequential(*layers)
    def forward(self, input:tensor) -> tensor:
        return self.net(input)
