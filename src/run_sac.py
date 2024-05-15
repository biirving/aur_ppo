#!/usr/bin/env python

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
from src.nets.base_cnns import PPOGaussianPolicy, PPOCritic, vitSACActor, vitSACCritic, SACGaussianPolicy, SACCritic
from src.utils.env_wrapper import EnvWrapper

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import collections
import copy

import torch.nn.functional as F
import numpy.random as npr

def behavioral_clone(envs, agent, bc_episodes=100):
    states, obs = envs.reset()
    bc_episodes = bc_episodes 
    bc_batch_size = 16
    expert_actions = []
    agent_actions = []
    obs_list = []
    j = 0
    update_epochs = 10 
    planner_bar = tqdm(total=bc_episodes)
    while j < bc_episodes:
        with torch.no_grad():
            true_action = envs.getNextAction()
            unscaled, scaled = agent.getActionFromPlan(true_action)
            expert_actions.append(unscaled.cpu().numpy())
            obs_to_add = torch.cat([obs, states.reshape(states.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])], dim=1)
        obs_list.append(obs_to_add.cpu().numpy())
        states, obs, reward, dones = envs.step(scaled, auto_reset=True)
        j += dones.sum().item()
        planner_bar.update(dones.sum().item())
    expert_tensor = torch.tensor(np.stack([a for a in expert_actions])).squeeze(dim=0)
    obs = torch.tensor(np.stack([o for o in obs_list]))
    flattened_expert = expert_tensor.view(expert_tensor.shape[0] * expert_tensor.shape[1], expert_tensor.shape[2])
    flattened_obs = obs.view(obs.shape[0] * obs.shape[1], obs.shape[2], obs.shape[3], obs.shape[4])
    total_bc_steps = flattened_expert.shape[0]
    inds = np.arange(total_bc_steps)
    for _ in range(update_epochs):
        np.random.shuffle(inds)
        for index in tqdm(range(0, total_bc_steps, bc_batch_size)):
            mb_inds = inds[index:index+bc_batch_size]
            # should update the agent directly
            agent.pretrain_update(flattened_obs[mb_inds].cuda(), flattened_expert[mb_inds].cuda())

def evaluate(global_step, num_eval_episodes, eval_envs, agent, eval_returns):
        eval_bar = tqdm(total=num_eval_episodes)
        s, o = eval_envs.reset()
        eval_ep = 0
        sum_r = 0

        eval_bar = tqdm(total=num_eval_episodes)
        while eval_ep < num_eval_episodes:
            u_a, a = agent.act(s.cuda(), o.cuda(), deterministic=True)
            s, o, r, d  = eval_envs.step(a, auto_reset=True)

            for i, rew in enumerate(r):
                if rew != 0:
                    print('reward', rew)
                eval_returns.add_value(i, rew)
            eval_ep += d.sum().item()
            eval_bar.update(d.sum().item())

            done_idxes = torch.nonzero(d).squeeze(1)
            best_return = float('-inf')
            shortest_length = float('inf')
            if done_idxes.shape[0] != 0:
                reset_states_, reset_obs_ = eval_envs.reset_envs(done_idxes)
                for j, idx in enumerate(done_idxes):
                    discounted_return, episode_length = eval_returns.calc_discounted_return(idx)
                    if discounted_return > best_return:
                        best_return = discounted_return
                        shortest_length = episode_length
                sum_r += best_return
        mean_r = sum_r / num_eval_episodes
        #writer.add_scalar("charts/eval_discounted_episodic_return", mean_r, global_step=global_step)

def sac(render, save_path=None, ac_kwargs=dict(), seed=0, 
        num_processes=1, steps_per_epoch=1000, epochs=4, replay_size=int(1e5), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=64, start_steps=10000, 
        update_after=1000, update_every=50, pretrain_episodes=20, bc_episodes=100, num_test_episodes=100, 
        max_ep_len=100,track=False, save_freq=1, gym_id=None, encoder_type='base', device=torch.device('cuda')):

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
    test_envs = EnvWrapper(num_processes, simulator, gym_id, env_config, planner_config)
    torch.set_num_threads(torch.get_num_threads())

    gamma = 0.99
    episodic_returns = store_returns(num_processes, gamma)
    eval_returns = store_returns(num_processes, gamma)
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

    agent = sacBullet() 

    if encoder_type == 'base':
        actor = SACGaussianPolicy().cuda()
        critic = SACCritic().cuda()
    elif encoder_type == 'equiv':
        actor = EquivariantSACActor().cuda()
        critic = EquivariantSACCritic().cuda()
    elif encoder_type == 'vit':
        actor = vitActor().cuda()
        critic = vitCritic().cuda()
    else:
        raise ValueError('Encoder type not found.')

    agent.initNet(actor, critic, encoder_type)

    if track:
        import wandb
        wandb.init(project='sac',entity='Aurelian',sync_tensorboard=True,config=None,name=gym_id + '_' + str(lr)) #+ '_' 
       # str(self.value_coeff) + '_' + str(elf.entropy_coeff) + '_' + str(self.clip_coeff) + '_' + str(self.num_minibatches),monitor_gym=True,save_code=True)
    writer = SummaryWriter(f"runs/{gym_id}")
    #writer.add_text(
    #"hyperparameters",
    #"|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{str(self.params_dict[key])}|" for key in self.params_dict])),
    #)


    torch.manual_seed(seed)
    np.random.seed(seed)
    
    #replay_buffer = QLearningBufferAug(replay_size)
    replay_buffer = QLearningBuffer(replay_size)

    #agent.train()

    # so this shit doesn't generalize?
    #pretrained_agent = torch.load('/scratch/irving.b/rl/close_loop_block_in_bowl_agent.pt')
    #pretrained_critic = torch.load('/scratch/irving.b/rl/close_loop_block_in_bowl_critic.pt')
    #agent.pi.load_state_dict(pretrained_agent)
    #agent.critic.load_state_dict(pretrained_critic)

    # using behavioral cloning before SAC training loop
    mse_envs = envs
    behavioral_clone(mse_envs, agent, bc_episodes)
    evaluate(0, num_test_episodes, test_envs, agent, eval_returns)
    sys.exit()

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

    agent.save_agent(gym_id, save_path)
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
    parser.add_argument('-sp', '--save_path', type=str, default='/scratch/irving.b/rl')
    parser.add_argument('-bc', '--bc_episodes', type=int, default=100)
    parser.add_argument('-et', '--encoder_type', type=str, default='base')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('training')
    sac(args.render, save_path=args.save_path, ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, num_processes=args.num_envs, epochs=args.epochs,
        track=args.track, gym_id=args.gym_id, bc_episodes=args.bc_episodes, pretrain_episodes=20,
        device=device, encoder_type=args.encoder_type)