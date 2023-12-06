from env_wrapper import EnvWrapper
import numpy as np

num_processes=10
num_eval_processes=5
simulator='pybullet'
env='close_loop_block_pulling'
env_config={'workspace': np.array([[ 0.25,  0.65],
       [-0.2 ,  0.2 ],
       [ 0.01,  0.25]]), 'max_steps': 100, 'obs_size': 128, 'fast_mode': True, 'action_sequence': 'pxyzr', 'render': False, 'num_objects': 2, 'random_orientation': True, 'robot': 'kuka', 'workspace_check': 'point', 'object_scale_range': (1, 1), 'hard_reset_freq': 1000, 'physics_mode': 'fast', 'view_type': 'camera_center_xyz', 'obs_type': 'pixel', 'view_scale': 1.5, 'transparent_bin': True}
planner_config={'random_orientation': True, 'dpos': 0.02, 'drot': 0.19634954084936207}
envs = EnvWrapper(num_processes, simulator, env, env_config, planner_config)

state, obs = envs.reset()
print(state)
act = envs.getNextAction()
eval_envs = EnvWrapper(num_eval_processes, simulator, env, env_config, planner_config)
what, what_2, what_3, what_4 = envs.step(act)
print(what)
print(what_3)
