import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

# similar amount of parameters
class base_encoder(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), out_dim=1024):
        super().__init__()
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.Conv2d(obs_shape[0], 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 64x64
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # 6x6
            nn.MaxPool2d(2),
            # 3x3
            nn.Conv2d(256, out_dim, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.conv(x)

# similar amount of parameters
class base_critic(nn.Module):
    def __init__(self, obs_shape=(1, 128, 128)):
        super().__init__()
        self.conv = base_encoder(obs_shape=obs_shape, out_dim=128)
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            torch.nn.Linear(128, 1)
        )
        self.apply(weights_init)

    def forward(self, obs):
        conv_out = self.conv(obs)
        out = self.critic(conv_out)
        return out 

# similar amount of parameters
class base_actor(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5):
        super().__init__()
        self.conv = base_encoder(obs_shape=obs_shape, out_dim=128)
        self.mean_linear = nn.Linear(128, action_dim)
        self.apply(weights_init)

    def forward(self, x):
        x = self.conv(x)
        mean = self.mean_linear(x)
        return mean
