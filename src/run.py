#!/usr/bin/env python
import sys, time, os

sys.path.append('../')
from src.trainer.ppoBulletTrainer import ppoBulletTrainer
from src.policies.ppoBullet import ppoBullet
from src.trainer.sacBulletTrainer import sacBulletTrainer
from src.trainer.ppoBulletTrainer import ppoBulletTrainer
from src.trainer.sacBulletOfflineTrainer import sacBulletOfflineTrainer
from src.policies.sacBullet import sacBullet
from src.utils.env_wrapper import EnvWrapper
from src.nets.base_cnns import vitSACActor, vitSACCritic, SACCritic, SACGaussianPolicy, PPOCritic, PPOGaussianPolicy, vitPPOActor
from src.nets.equiv import EquivariantActor, EquivariantCritic, EquivariantSACCritic, EquivariantSACActor
import numpy as np
import torch

import os, sys, argparse, time

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--gym_id', type=str, help='Id of the environment that we will use', default='close_loop_block_in_bowl')
    parser.add_argument('-render', '--render', type=str2bool, help='Whether or not to render the environment', default=False, nargs='?', const=False)
    parser.add_argument('-num_processes', '--num_processes', type=int, help='Number of processes', default=5)
    parser.add_argument('-track', '--track', type=str2bool, help='Track the rewards', default=False, nargs='?', const=False)
    parser.add_argument('-sp', '--save_path', type=str, default='/scratch/irving.b/rl')
    parser.add_argument('-epsp', '--episode_save_path', type=str, default='/scratch/irving.b/rl/episodes/')

    parser.add_argument('-bc', '--bc_episodes', type=int, help='Number of episodes for behavioral cloning', default=100)
    parser.add_argument('-pte', '--pretrain_episodes', type=int, help='Number of pretraining episodes', default=500)
    parser.add_argument('-alg', '--algorithm', type=str, help='Algorithm to use for training', default='sac')
    parser.add_argument('-g', '--gamma', type=float, default=0.99)
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-tre', '--training_episodes', type=int, default=1000)
    parser.add_argument('-rid', '--run_id', type=str, help='Run identification number', default=0)
    parser.add_argument('-et', '--encoder_type', type=str, default='base')

    parser.add_argument('-mgn', '--max_grad_norm', type=float, help='the maximum norm for the gradient clipping', default=0.5)
    parser.add_argument('-tkl', '--target_kl',type=float, help='The KL divergence that we will not exceed', default=None)
    parser.add_argument('-na', '--norm_adv', type=bool, help='Normalize advantage estimates', default=True)
    parser.add_argument('-actor_lr', '--actor_lr', type=float, help='Learning rate for actor', default=1e-3)
    parser.add_argument('-critic_lr', '--critic_lr', type=float, help='Learning rate for critic', default=1e-3)
    parser.add_argument('-cf', '--clip_coeff', type=float, help="the surrogate clipping coefficient",  default=0.2)
    parser.add_argument('-ec', '--entropy_coeff', type=float, help='Coefficient for entropy', default=0.01)
    parser.add_argument('-vf', '--value_coeff', type=float, help='Coefficient for values', default=0.5)
    parser.add_argument('-nm', '--num_minibatches', type=int, help='Number of minibatches', default=4)
    parser.add_argument('-expw', '--expert_weight', type=float, help='How much do we want the expert trajectory to contribute?', default=0.9)

    parser.add_argument('-mxp', '--mixed_expert_episodes', type=int, help='Number of mixed expert episodes for offline training', default=10000)
    args = parser.parse_args()

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
            'render': args.render, 
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
           # 'dists':False}
    planner_config={'random_orientation': True, 'dpos': dpos, 'drot': drot}
    encoder_type=args.encoder_type
    algo = args.algorithm

    if algo == 'sac' or algo == 'sacoffline':
        if encoder_type == 'base':
            actor = SACGaussianPolicy().cuda()
            critic = SACCritic().cuda()
        elif encoder_type == 'equiv':
            actor = EquivariantSACActor().cuda()
            critic = EquivariantSACCritic().cuda()
        elif encoder_type == 'vit':
            actor = vitSACActor().cuda()
            critic = vitSACCritic().cuda()
        agent = sacBullet()
        if algo == 'sac':
            trainer = sacBulletTrainer(agent, num_processes=args.num_processes, 
                track=args.track, pretrain_episodes=args.pretrain_episodes, bc_episodes=args.bc_episodes,
                save_path=args.save_path, run_id=args.run_id)
            trainer.run(simulator, env_config, planner_config, args.gym_id, actor, critic, encoder_type)
        else:
            trainer = sacBulletOfflineTrainer(agent, num_processes=args.num_processes, 
                track=args.track, pretrain_episodes=args.pretrain_episodes, bc_episodes=args.bc_episodes,
                save_path=args.save_path, run_id=args.run_id, episode_save_path=args.episode_save_path, mixed_expert_episodes=args.mixed_expert_episodes)
            trainer.run(simulator, env_config, planner_config, args.gym_id, actor, critic, encoder_type)
    elif algo == 'ppo':
        if encoder_type == 'base':
            actor = PPOGaussianPolicy().cuda()
            critic = PPOCritic().cuda()
        elif encoder_type == 'equiv':
            pass
            #actor = EquivariantSACActor().cuda()
            #critic = EquivariantSACCritic().cuda()
        elif encoder_type == 'vit':
            actor = vitPPOActor().cuda()
            critic = vitSACritic().cuda()
        agent = ppobullet(num_processes=args.num_processes, actor_lr=args.actor_lr, critic_lr=args.critic_lr)
        trainer = ppobullettrainer(agent, num_processes=args.num_processes, 
                track=args.track, pretrain_episodes=args.pretrain_episodes, run_id=args.run_id)
        trainer.run(simulator, env_config, planner_config, args.gym_id, actor, critic, encoder_type)
