import torch
import torch.nn as nn
from torch.distributions import Normal
from src.nets.nets import SACGaussianPolicyBase, PPOGaussianPolicyBase
from transformers import AutoImageProcessor, ViTModel, ViTConfig

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
    def __init__(self, obs_shape=(2, 128, 128)):
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


# similar amount of parameters
class SACCritic(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5):
        super().__init__()
        self.state_conv_1 = base_encoder(obs_shape, 128)

        # Q1
        self.critic_fc_1 = torch.nn.Sequential(
            torch.nn.Linear(128+action_dim, 128),
            nn.ReLU(inplace=True),
            torch.nn.Linear(128, 1)
        )

        # Q2
        self.critic_fc_2 = torch.nn.Sequential(
            torch.nn.Linear(128+action_dim, 128),
            nn.ReLU(inplace=True),
            torch.nn.Linear(128, 1)
        )

        self.apply(weights_init)

    def forward(self, obs, act):
        conv_out = self.state_conv_1(obs)
        out_1 = self.critic_fc_1(torch.cat((conv_out, act), dim=1))
        out_2 = self.critic_fc_2(torch.cat((conv_out, act), dim=1))
        return out_1, out_2

class SACGaussianPolicy(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5):
        super().__init__()
        self.conv = base_encoder(obs_shape, 128)
        self.mean_linear = nn.Linear(128, action_dim)
        self.log_std_linear = nn.Linear(128, action_dim)

        self.apply(weights_init)

    def forward(self, x):
        x = self.conv(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

class PPOGaussianPolicy(PPOGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5):
        super().__init__()
        self.conv = base_encoder(obs_shape, 128)
        self.mean_linear = nn.Linear(128, action_dim)
        self.log_std_linear = nn.Linear(128, action_dim)

    def forward(self, x):
        x = self.conv(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

class PPOCritic(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5):
        super().__init__()
        self.conv = base_encoder(obs_shape, 128)
        self.critic = torch.nn.Sequential(
				torch.nn.Linear(128, 128),
				nn.ReLU(inplace=True),
				torch.nn.Linear(128, 1))
    def forward(self, x):
        x = self.conv(x)
        return self.critic(x)

class vitWrapper(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), pretrained=True):
        super().__init__()
        config = ViTConfig()
        config.num_channels = obs_shape[0]
        config.image_size = obs_shape[1]
        self.model = ViTModel(config)
        
        if pretrained:
            pretrained_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
            # Prepare new state dict from pretrained model
            new_state_dict = {}
            for name, param in pretrained_model.named_parameters():
                if name in self.model.state_dict() and param.size() == self.model.state_dict()[name].size():
                    new_state_dict[name] = param
                else:
                    print(f"Skipping {name} due to size mismatch or it's missing in the custom model.")
            self.model.load_state_dict(new_state_dict, strict=False)

    def forward(self, x):
        return self.model(x)

class vitActor(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5):
        super().__init__()
        self.encoder = vitWrapper(obs_shape)
        self.mean_linear = nn.Linear(768, action_dim)
        self.log_std_linear = nn.Linear(768, action_dim)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0., std=0.1)
            torch.nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        encoded = self.encoder(x)
        pooler_output = encoded.pooler_output
        mean = self.mean_linear(pooler_output)
        log_std = self.log_std_linear(pooler_output)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

# similar amount of parameters
class vitCritic(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5):
        super().__init__()
        self.encoder=vitWrapper()

        # Q1
        self.critic_fc_1 = torch.nn.Sequential(
            torch.nn.Linear(768+action_dim, 128),
            nn.ReLU(inplace=True),
            torch.nn.Linear(128, 1)
        )

        # Q2
        self.critic_fc_2 = torch.nn.Sequential(
            torch.nn.Linear(768+action_dim, 128),
            nn.ReLU(inplace=True),
            torch.nn.Linear(128, 1)
        )

        self.apply(weights_init)

    def forward(self, obs, act):
        encoder_out = self.encoder(obs)
        encoder_out = encoder_out.pooler_output
        out_1 = self.critic_fc_1(torch.cat((encoder_out, act), dim=1))
        out_2 = self.critic_fc_2(torch.cat((encoder_out, act), dim=1))
        return out_1, out_2