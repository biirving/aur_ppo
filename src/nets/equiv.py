from e2cnn import nn, gspaces
import torch
from torch.distributions import Normal
import sys
sys.path.append('..')
from src.nets.nets import SACGaussianPolicyBase

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class EquivariantEncoder128(torch.nn.Module):
    def __init__(self, obs_channel=1, n_out=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_channel
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.c4_act, obs_channel * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.regular_repr]), inplace=True),

            nn.R2Conv(nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            # 1x1
        )

    def forward(self, geo):
        return self.conv(geo)


class EquivariantActor(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.n_rho1 = 2 if N==2 else 1
        self.conv = torch.nn.Sequential(
            EquivariantEncoder128(self.obs_channel, n_hidden, initialize, N),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      # mixed representation including action_dim trivial representations (for the std of all actions),
                      # (action_dim-2) trivial representations (for the mu of invariant actions),
                      # and 1 standard representation (for the mu of equivariant actions)
                      nn.FieldType(self.c4_act, self.n_rho1 * [self.c4_act.irrep(1)] + (action_dim*2-2) * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        log_std = conv_out[:, self.action_dim:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std


class EquivariantSACActor(SACGaussianPolicyBase):
    """
    Equivariant SAC's equivariant actor
    """
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, enc_id=1):
        super().__init__()
        assert obs_shape[1] in [128, 64]
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.c4_act = gspaces.Rot2dOnR2(N)
        enc = getEnc(obs_shape[1], enc_id)
        self.n_rho1 = 2 if N==2 else 1
        self.conv = torch.nn.Sequential(
            enc(self.obs_channel, n_hidden, initialize, N),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      # mixed representation including action_dim trivial representations (for the std of all actions),
                      # (action_dim-2) trivial representations (for the mu of invariant actions),
                      # and 1 standard representation (for the mu of equivariant actions)
                      nn.FieldType(self.c4_act, self.n_rho1 * [self.c4_act.irrep(1)] + (action_dim*2-2) * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        log_std = conv_out[:, self.action_dim:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

class EquivariantCritic(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), n_hidden=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        # N is which cyclic subgroup we are using
        # in this case, we are using C_4
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.img_conv = EquivariantEncoder128(self.obs_channel, n_hidden, initialize, N)
        self.n_rho1 = 2 if N==2 else 1

        self.critic = torch.nn.Sequential(
            # mixed representation including n_hidden regular representations (for the state),
            # (action_dim-2) trivial representations (for the invariant actions)
            # and 1 standard representation (for the equivariant actions)
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    # so in this case, the critic is a Q function ?
    def forward(self, obs):
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        out = self.critic(conv_out)
        return out 

def getEnc(obs_size, enc_id):
    assert obs_size in [128]
    assert enc_id in [1]
    return EquivariantEncoder128

class EquivariantSACCritic(torch.nn.Module):
    """
    Equivariant SAC's invariant critic
    """
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, enc_id=1):
        super().__init__()
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.c4_act = gspaces.Rot2dOnR2(N)
        enc = getEnc(obs_shape[1], enc_id)
        self.img_conv = enc(self.obs_channel, n_hidden, initialize, N)
        self.n_rho1 = 2 if N==2 else 1

        self.critic_1 = torch.nn.Sequential(
            # mixed representation including n_hidden regular representations (for the state),
            # (action_dim-2) trivial representations (for the invariant actions)
            # and 1 standard representation (for the equivariant actions)
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim-2) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.critic_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim-2) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, obs, act):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)
        n_inv = inv_act.shape[1]
        cat = torch.cat((conv_out.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1)), dim=1)
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + n_inv * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]))
        out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
        out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
        return out1, out2
    
class EquivariantAWACCritic(torch.nn.Module):
    """
    Equivariant AWAC's invariant critic
    """
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, enc_id=1):
        super().__init__()
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.c4_act = gspaces.Rot2dOnR2(N)
        enc = getEnc(obs_shape[1], enc_id)
        self.img_conv = enc(self.obs_channel, n_hidden, initialize, N)
        self.n_rho1 = 2 if N==2 else 1

        self.critic = torch.nn.Sequential(
            # mixed representation including n_hidden regular representations (for the state),
            # (action_dim-2) trivial representations (for the invariant actions)
            # and 1 standard representation (for the equivariant actions)
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr] + (action_dim-2) * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, obs, act):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)
        n_inv = inv_act.shape[1]
        cat = torch.cat((conv_out.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1)), dim=1)
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.c4_act, self.n_hidden * [self.c4_act.regular_repr] + n_inv * [self.c4_act.trivial_repr] + self.n_rho1*[self.c4_act.irrep(1)]))
        out = self.critic(cat_geo).tensor.reshape(batch_size, 1)
        return out


# a class for the equivariant vision transformer
class equivariantVitActor(SACGaussianPolicyBase):
    def __init__(self):
        pass  