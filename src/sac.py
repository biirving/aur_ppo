# sanity check
# open ai SAC implementation, with a few changes

import numpy as np
import torch
import time
import sys

sys.path.append('/work/nlp/b.irving/aur_ppo/src')
import models.sac_core as core
from utils.str2bool import str2bool
from env_wrapper_2 import EnvWrapper

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import collections
import copy

import torch.nn.functional as F

#sys.path.append('/home/benjamin/Desktop/ml/BulletArm/bulletarm_baselines')
#from bulletarm_baselines.equi_rl.agents.sac import SAC
#from bulletarm_baselines.equi_rl.networks.sac_net import SACCritic, SACGaussianPolicy
#from bulletarm_baselines.equi_rl.networks.equivariant_sac_net import EquivariantSACActor, EquivariantSACCritic, EquivariantSACActorDihedral, EquivariantSACCriticDihedral

ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done step_left expert')
from scipy.ndimage import affine_transform


import numpy.random as npr

def get_image_transform(theta, trans, pivot=(0, 0)):
    """Compute composite 2D rigid transformation matrix."""
    # Get 2D rigid transformation matrix that rotates an image by theta (in
    # radians) around pivot (in pixels) and translates by trans vector (in
    # pixels)
    pivot_t_image = np.array([[1., 0., -pivot[0]], [0., 1., -pivot[1]],
                              [0., 0., 1.]])
    image_t_pivot = np.array([[1., 0., pivot[0]], [0., 1., pivot[1]],
                              [0., 0., 1.]])
    transform = np.array([[np.cos(theta), -np.sin(theta), trans[0]],
                          [np.sin(theta), np.cos(theta), trans[1]], [0., 0., 1.]])
    return np.dot(image_t_pivot, np.dot(transform, pivot_t_image))

def get_random_image_transform_params(image_size):
    theta = np.random.random() * 2*np.pi
    trans = np.random.randint(0, image_size[0]//10, 2) - image_size[0]//20
    pivot = (image_size[1] / 2, image_size[0] / 2)
    return theta, trans, pivot

def perturb(current_image, next_image, dxy, set_theta_zero=False, set_trans_zero=False):
    image_size = current_image.shape[-2:]

    # Compute random rigid transform.
    theta, trans, pivot = get_random_image_transform_params(image_size)
    if set_theta_zero:
        theta = 0.
    if set_trans_zero:
        trans = [0., 0.]
    transform = get_image_transform(theta, trans, pivot)
    transform_params = theta, trans, pivot

    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_dxy = rot.dot(dxy)
    rotated_dxy = np.clip(rotated_dxy, -1, 1)

    # Apply rigid transform to image and pixel labels.
    current_image = affine_transform(current_image, np.linalg.inv(transform), mode='nearest', order=1)
    if next_image is not None:
        next_image = affine_transform(next_image, np.linalg.inv(transform), mode='nearest', order=1)

    return current_image, next_image, rotated_dxy, transform_params
    

def augmentTransitionSO2(d):
    obs, next_obs, dxy, transform_params = perturb(d.obs[0].copy(),
                                                   d.next_obs[0].copy(),
                                                   d.action[1:3].copy(),
                                                   set_trans_zero=True)
    obs = obs.reshape(1, *obs.shape)
    next_obs = next_obs.reshape(1, *next_obs.shape)
    action = d.action.copy()
    action[1] = dxy[0]
    action[2] = dxy[1]
    return ExpertTransition(d.state, obs, action, d.reward, d.next_state,
                            next_obs, d.done, d.step_left, d.expert)
class QLearningBuffer:
    def __init__(self, size):
        self._storage = []
        self._max_size = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def __getitem__(self, key):
        return self._storage[key]

    def __setitem__(self, key, value):
        self._storage[key] = value

    def add(self, data):
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._max_size

    def sample(self, batch_size):
        batch_indexes = npr.choice(self.__len__(), batch_size).tolist()
        batch = [self._storage[idx] for idx in batch_indexes]
        return batch

    def getSaveState(self):
        return {
            'storage': self._storage,
            'max_size': self._max_size,
            'next_idx': self._next_idx
        }

    def loadFromState(self, save_state):
        self._storage = save_state['storage']
        self._max_size = save_state['max_size']
        self._next_idx = save_state['next_idx']
class QLearningBufferAug(QLearningBuffer):
    def __init__(self, size, aug_n=4):
        super().__init__(size)
        self.aug_n = aug_n

    def add(self, transition: ExpertTransition):
        super().add(transition)
        for _ in range(self.aug_n):
            super().add(augmentTransitionSO2(transition))

def normalizeTransition(d: ExpertTransition):
    obs = np.clip(d.obs, 0, 0.32)
    obs = obs/0.4*255
    obs = obs.astype(np.uint8)

    next_obs = np.clip(d.next_obs, 0, 0.32)
    next_obs = next_obs/0.4*255
    next_obs = next_obs.astype(np.uint8)

    return ExpertTransition(d.state, obs, d.action, d.reward, d.next_state, next_obs, d.done, d.step_left, d.expert)

class store_returns():
    def __init__(self, num_envs, gamma):
        self.gamma = gamma
        self.env_returns = [[] for _ in range(num_envs)]

    def add_value(self, i, reward):
        self.env_returns[i].append(reward)

    def calc_discounted_return(self, i):
        len_episode = len(self.env_returns[i])
        R = 0
        for r in self.env_returns[i][::-1]:
            R = r + self.gamma * R
            self.env_returns[i] = []
        return R, len_episode

def sac(render, ac_kwargs=dict(), seed=0, 
        num_processes=1, steps_per_epoch=1000, epochs=4, replay_size=int(1e5), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=64, start_steps=10000, 
        update_after=1000, update_every=50, pretrain_episodes=20, num_test_episodes=10, 
        max_ep_len=100,track=False, save_freq=1, gym_id=None, device=torch.device('cpu')):

    simulator='pybullet'
    gamma = 0.99
    lr = 1e-3
    dpos = 0.05
    drot = np.pi/8
    obs_type='pixel'
    action_sequence=5
    workspace_size=0.3
    workspace = np.asarray([[0.45-workspace_size/2, 0.45+workspace_size/2],
                        [0-workspace_size/2, 0+workspace_size/2],
                        [0.01, 0.25]])
    env_config={'workspace': workspace, 'max_steps': 100, 
            'obs_size': 128, 
            'fast_mode': True, 
            'action_sequence': 'pxyzr', 
            'render': render, 
            'num_objects': 2, 
            'random_orientation': True, 
            'robot': 'kuka', 
            'workspace_check': 'point', 
            'object_scale_range': (1, 1), 
            'hard_reset_freq': 1000, 
            'physics_mode': 'fast', 
            'view_type': 'camera_center_xyz', 
            'obs_type': 'pixel', 
            'view_scale': 1.5, 
            'transparent_bin': True}
    planner_config={'random_orientation': True, 'dpos': dpos, 'drot': drot}
    envs = EnvWrapper(num_processes, simulator, gym_id, env_config, planner_config)
    test_envs = EnvWrapper(num_processes, simulator, gym_id, env_config, planner_config)
    torch.set_num_threads(torch.get_num_threads())


    gamma = 0.99
    episodic_returns = store_returns(num_processes, gamma)
    lr = 1e-3
    dpos = 0.05
    drot = np.pi/8
    obs_type='pixel'
    action_sequence=5
    obs_channel=2
    crop_size = 128
    equi_n = 8
    n_hidden = 128 
    initialize=True


    agent = core.MLPActorCritic()

    if track:
        import wandb
        wandb.init(project='sac',entity='Aurelian',sync_tensorboard=True,config=None,name=gym_id + '_' + str(lr)) #+ '_' 
       # str(self.value_coeff) + '_' + str(self.entropy_coeff) + '_' + str(self.clip_coeff) + '_' + str(self.num_minibatches),monitor_gym=True,save_code=True)
    writer = SummaryWriter(f"runs/{gym_id}")
    #writer.add_text(
    #"hyperparameters",
    #"|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{str(self.params_dict[key])}|" for key in self.params_dict])),
    #)


    torch.manual_seed(seed)
    np.random.seed(seed)
    
    replay_buffer = QLearningBufferAug(replay_size)

    def get_action(s, o, deterministic=False):
        with torch.no_grad():
            return agent.act(s.to(device), o.to(device),
                      deterministic)

    def test_agent():
        ep_ret, ep_len = 0, 0
        for j in tqdm(range(num_test_episodes)):
            s, o = test_envs.reset()
            d, ep_ret, ep_len = False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                _, a = get_action(s, o, False)
                s, o, r, d = test_envs.step(a)
                ep_ret += r
                ep_len += 1
        return ep_ret, ep_len

    agent.train()

    counter = 0
    update_counter = 0
    if pretrain_episodes > 0:
        planner_envs = envs

        planner_num_process = num_processes
        j = 0
        states, obs = planner_envs.reset()
        s = 0

        planner_bar = tqdm(total=pretrain_episodes)
        while j < pretrain_episodes:
            plan_actions = planner_envs.getNextAction()
            with torch.no_grad():
                planner_actions_star_idx, planner_actions_star = agent.getActionFromPlan(plan_actions)
                states_, obs_, rewards, dones = planner_envs.step(planner_actions_star, auto_reset=True)

            for i in range(planner_num_process):
                transition = ExpertTransition(states[i].numpy(), obs[i].numpy(), planner_actions_star_idx[i].numpy(),
                                              rewards[i].numpy(), states_[i].numpy(), obs_[i].numpy(), 
                                              dones[i],np.array(100), np.array(1))
                transition = normalizeTransition(transition)
                replay_buffer.add(transition)
                counter += 1
            states = copy.copy(states_)
            obs = copy.copy(obs_)
            j += dones.sum().item()
            s += rewards.sum().item()
            planner_bar.set_description('{:.3f}/{}, AVG: {:.3f}'.format(s, j, float(s) / j if j != 0 else 0))
            planner_bar.update(dones.sum().item())
            #if expert_aug_n > 0:
            #    augmentBuffer(replay_buffer, buffer_aug_type, expert_aug_n)
    pretrain_step = counter // batch_size
    pretrain_step = 0
    if pretrain_step > 0:
        for i in tqdm(range(pretrain_step)):
            batch = replay_buffer.sample(batch_size)
            agent.update(batch)

    total_steps = steps_per_epoch * epochs
    start_time = time.time()

    s, o  = envs.reset()
    ep_ret, ep_len = torch.zeros(num_processes), torch.zeros(num_processes) 
    last_ret, last_len = 0, 0

    # Main loop: collect experience in env and update/log each epoch
    replay_len=0
    for t in tqdm(range(total_steps)):
        
        u_a, a = get_action(s, o)

        envs.stepAsync(a, auto_reset=False)

        if len(replay_buffer) >= 100:
            batch = replay_buffer.sample(batch_size)
            agent.update(batch)

        s2, o2, r, d  = envs.stepWait()
        for i, rew in enumerate(r):
            episodic_returns.add_value(i, rew)
        ep_ret += r 
        ep_len += torch.ones(num_processes) 

        done_idxes = torch.nonzero(d).squeeze(1)

        if done_idxes.shape[0] != 0:
            reset_states_, reset_obs_ = envs.reset_envs(done_idxes)
            for j, idx in enumerate(done_idxes):
                s2[idx] = reset_states_[j]
                o2[idx] = reset_obs_[j]
                discounted_return, episode_length = episodic_returns.calc_discounted_return(idx)
                writer.add_scalar("charts/discounted_episodic_return", discounted_return, global_step=t)
                writer.add_scalar("charts/episodic_length", episode_length, global_step=t)
                ep_ret[idx] = 0 
                ep_len[idx] = 0

        for i in range(num_processes):
            transition = ExpertTransition(s[i].numpy(), o[i].numpy(), u_a[i].numpy(), r[i].numpy(), s2[i].numpy(), o2[i].numpy(), d[i].numpy(), np.array(100), 0)
            transition = normalizeTransition(transition)
            replay_buffer.add(transition)
            replay_len+=1

        o = copy.copy(o2)
        s = copy.copy(s2)

        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

    
    envs.close()
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('-id', '--gym_id', type=str, help='Id of the environment that we will use', default='close_loop_block_reaching')
    parser.add_argument('--hid', type=int, default=1024)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('-tr', '--track', type=str2bool, help='Track the performance of the environment', nargs='?', const=False, default=False)
    parser.add_argument('-ne', '--num_envs', type=int, default=1)
    parser.add_argument('-re', '--render', type=str2bool, nargs='?', const=False, default=False)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('training')
    sac(args.render, ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, num_processes=args.num_envs, epochs=args.epochs,
        track=args.track, gym_id=args.gym_id, pretrain_episodes=100, device=device)