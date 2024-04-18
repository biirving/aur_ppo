import sys, time, os
sys.path.append('../')
from src.trainer.ppoBulletTrainer import ppoBulletTrainer
from src.policies.ppoBullet import ppoBullet
from src.utils.env_wrapper import EnvWrapper
from src.nets.base_cnns import vitActor, vitCritic, SACCritic, SACGaussianPolicy, PPOCritic, PPOGaussianPolicy
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
    parser.add_argument('-id', '--gym_id', type=str, help='Id of the environment that we will use', default='close_loop_block_reaching')
    parser.add_argument('-render', '--render', type=str2bool, help='Whether or not to render the environment', default=False, nargs='?', const=False)
    parser.add_argument('-num_processes', '--num_processes', type=int, help='Number of processes', default=5)
    parser.add_argument('-track', '--track', type=str2bool, help='Track the rewards', default=False, nargs='?', const=False)
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
    planner_config={'random_orientation': True, 'dpos': dpos, 'drot': drot}

    agent = ppoBullet(num_processes=args.num_processes)
    trainer = ppoBulletTrainer(agent, num_processes=args.num_processes, track=args.track)
    trainer.run(simulator, env_config, planner_config, args.gym_id, PPOGaussianPolicy().cuda(), PPOCritic().cuda())
