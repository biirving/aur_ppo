import numpy as np
import torch
import time
import sys

sys.path.append('../')
import src.models.sac_core as core
from src.policies.offlineSACBullet import offlineSACBullet 
from src.utils.str2bool import str2bool
from src.utils.buffers import QLearningBuffer, QLearningBufferAug
from src.utils.misc import ExpertTransitionOffline, normalizeTransitionOffline, store_returns
from src.nets.equiv import EquivariantActor, EquivariantCritic, EquivariantSACCritic, EquivariantSACActor
from src.nets.base_cnns import PPOGaussianPolicy, PPOCritic, vitActor, vitCritic, SACGaussianPolicy, SACCritic
from src.utils.env_wrapper import EnvWrapper
from src.policies.offlineSACBullet import offlineSACBullet

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
    
def evaluate(global_step, num_eval_episodes, eval_envs, agent, writer, num_processes, gamma):
        eval_bar = tqdm(total=num_eval_episodes)
        s, o = eval_envs.reset()
        eval_ep = 0
        sum_r = 0
        eval_returns = store_returns(num_processes, gamma)

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
                writer.add_scalar("charts/eval_discounted_episodic_return", best_return, global_step=global_step)

def sac_offline(render, save_path=None, ac_kwargs=dict(), seed=0, 
        num_processes=1, steps_per_epoch=1000, epochs=4, replay_size=int(1e5), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=64, start_steps=10000, 
        update_after=1000, update_every=50, bc_episodes=20, training_episodes=1000,
        max_ep_len=100, num_test_episodes=100, track=False, save_freq=1, offline_updates=100,
        gym_id=None, device=torch.device('cuda'), encoder_type='base', episode_save_path=None, run_id=None):

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

    agent = offlineSACBullet() 
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

    agent.initNet(actor, critic)
    training_agent = agent

    if track:
        import wandb
        wandb.init(project='sac',entity='Aurelian',sync_tensorboard=True,config=None,name=gym_id + '_' + str(lr)) #+ '_' 
    writer = SummaryWriter(f"runs/{gym_id}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    replay_buffer = QLearningBuffer(replay_size, transition_type='offline')

    agent.train()

    # using behavioral cloning before SAC training loop
    mse_envs = envs
    behavioral_clone(mse_envs, agent, bc_episodes=bc_episodes)
    evaluate(0, num_test_episodes, test_envs, agent, writer, num_processes, gamma)

    counter = 0
    if training_episodes > 0:
        training_envs = envs
        training_num_processes = num_processes
        j = 0
        states, obs = training_envs.reset()
        s = 0
        training_bar = tqdm(total=training_episodes)
        # should have some bad episodes
        
        # every 10th episode should be a random policy
        while j < training_episodes:
            train_actions = training_envs.getNextAction()
            with torch.no_grad():
                training_actions_star_idx, training_actions_star = training_agent.getActionFromPlan(train_actions)
                random_actions_star_idx, random_actions_star =  training_agent.act(states.to(device), obs.to(device), deterministic=False)
                if counter % 10 == 0:
                    states_, obs_, rewards, dones = training_envs.step(random_actions_star, auto_reset=True)
                else:
                    states_, obs_, rewards, dones = training_envs.step(training_actions_star, auto_reset=True)
            for i in range(training_num_processes):
                transition = ExpertTransitionOffline(states[i].numpy(), obs[i].numpy(), random_actions_star_idx[i].numpy(),
                                              rewards[i].numpy(), states_[i].numpy(), obs_[i].numpy(), 
                                              dones[i], np.array(100), np.array(1), training_actions_star_idx[i].numpy())
                transition = normalizeTransitionOffline(transition)
                replay_buffer.add(transition)
                counter += 1
            states = copy.copy(states_)
            obs = copy.copy(obs_)
            j += dones.sum().item()
            s += rewards.sum().item()
            training_bar.set_description('{:.3f}/{}, AVG: {:.3f}'.format(s, j, float(s) / j if j != 0 else 0))
            training_bar.update(dones.sum().item())

    if episode_save_path is not None:
        replay_buffer.save_buffer(episode_save_path, run_id)
    start_time = time.time()
    offline_updates = counter // batch_size
    assert offline_updates > 0
    for i in tqdm(range(offline_updates)):
        batch = replay_buffer.sample(batch_size)
        agent.update(batch)
        # evaluate every 100 updates
        if i % 100 == 0:
            evaluate(i, num_test_episodes, test_envs, agent, eval_returns, writer, num_processes, gamma)

    agent.save_agent(gym_id, save_path)
    envs.close()
    test_envs.close()
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
    parser.add_argument('-et', '--encoder_type', type=str, default='base')
    parser.add_argument('-bc', '--bc_episodes', type=int, default=100)
    parser.add_argument('-tre', '--training_episodes', type=int, default=1000)
    parser.add_argument('-rid', '--run_id', type=str, help='Run identification number', default=0)
    parser.add_argument('-epsp', '--episode_save_path', type=str, default='/scratch/irving.b/rl/episodes/')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('training')
    sac_offline(args.render, save_path=args.save_path, training_episodes=args.training_episodes, ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, num_processes=args.num_envs, epochs=args.epochs,
        track=args.track, gym_id=args.gym_id, bc_episodes=args.bc_episodes, device=device, encoder_type=args.encoder_type, episode_save_path=args.episode_save_path, run_id=args.run_id)