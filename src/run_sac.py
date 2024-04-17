# sanity check
# open ai SAC implementation, with a few changes

import numpy as np
import torch
import time
import sys

sys.path.append('../')
import src.models.sac_core as core
from src.policies.sacBullet import sacBullet 
from src.utils.str2bool import str2bool
from src.utils.buffers import QLearningBuffer, QLearningBufferAug
from src.utils.misc import ExpertTransition, normalizeTransition, store_returns
from src.nets.equiv import EquivariantActor, EquivariantCritic, EquivariantSACCritic, EquivariantSACActor
from src.nets.base_cnns import PPOGaussianPolicy, PPOCritic, vitActor, vitCritic, SACGaussianPolicy, SACCritic
from src.utils.env_wrapper import EnvWrapper

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import collections
import copy

import torch.nn.functional as F
import numpy.random as npr



def sac(render, ac_kwargs=dict(), seed=0, 
        num_processes=1, steps_per_epoch=1000, epochs=4, replay_size=int(1e5), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=64, start_steps=10000, 
        update_after=1000, update_every=50, pretrain_episodes=20, num_test_episodes=10, 
        max_ep_len=100,track=False, save_freq=1, gym_id=None, device=torch.device('cuda')):

    # the way the experiment is run should also be cleaned up
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
    env_config={'workspace': workspace, 'max_steps': 50, 
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
    #test_envs = EnvWrapper(num_processes, simulator, gym_id, env_config, planner_config)
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

    #agent = core.aur_sac()
    agent = sacBullet() 
    actor = EquivariantSACActor().cuda()
    critic = EquivariantSACCritic().cuda()
    agent.initNet(actor, critic)

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


    """
    def test_agent():
        ep_ret, ep_len = 0, 0
        for j in tqdm(range(num_test_episodes)):
            s, o = test_envs.reset()
            d, ep_ret, ep_len = False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                _, a = agent.act(s, o, True)
                s, o, r, d = test_envs.step(a)
                ep_ret += r
                ep_len += 1
        return ep_ret, ep_len
    """

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

    replay_len=0
    for t in tqdm(range(total_steps)):
        u_a, a = agent.act(s.to(device), o.to(device), deterministic=False)

        envs.stepAsync(a, auto_reset=False)

        if len(replay_buffer) >= 100:
            batch = replay_buffer.sample(batch_size)
            agent.update(batch)

        s2, o2, r, d  = envs.stepWait()
        for i, rew in enumerate(r):
            if rew != 0:
                print('reward', rew)
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
    parser.add_argument('-id', '--gym_id', type=str, help='Id of the environment that we will use', default='close_loop_block_in_bowl')
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
        track=args.track, gym_id=args.gym_id, pretrain_episodes=20, device=device)