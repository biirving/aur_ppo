from env_wrapper import EnvWrapper
from tqdm import tqdm
import torch
import numpy as np
from nets.base_cnns import base_critic, base_actor
from models import robot_actor_critic

what = torch.tensor([0, 0])
if what.sum():
       print('True')
"""
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
num_processes=1
num_eval_processes=5
simulator='pybullet'
env='close_loop_block_stacking'
env_config={'workspace': np.array([[ 0.25,  0.65],
       [-0.2 ,  0.2 ],
       [ 0.01,  0.25]]), 'max_steps': 1024, 'obs_size': 128, 'fast_mode': True, 'action_sequence': 'pxyzr', 'render': False, 'num_objects': 2, 'random_orientation': True, 'robot': 'kuka', 'workspace_check': 'point', 'object_scale_range': (1, 1), 'hard_reset_freq': 1000, 'physics_mode': 'fast', 'view_type': 'camera_center_xyz', 'obs_type': 'pixel', 'view_scale': 1.5, 'transparent_bin': True}
planner_config={'random_orientation': True, 'dpos': 0.02, 'drot': 0.19634954084936207}
envs = EnvWrapper(num_processes, simulator, env, env_config, planner_config)

device = torch.device('cuda')
test = robot_actor_critic(device, True)
episodes = store_returns(num_processes, 0.99)
state, obs = envs.reset()
for index in tqdm(range(10000)):
       act = envs.getNextAction()
       unscaled, scaled = test.getActionFromPlan(act)
       next_state, next_obs, reward, done = envs.step(scaled, auto_reset=True)
       for i, rew in enumerate(reward):
              episodes.add_value(i, rew)
       for i, d in enumerate(done):
              if d:
                     discounted_return, episode_length = episodes.calc_discounted_return(i)
                     print(discounted_return)
                     print(episode_length)
"""

"""
device = torch.device('cuda')
test = robot_actor_critic(device, True)

state = torch.randn((5))
obs = torch.randn((5, 1, 128, 128))

actions, act, _, _, _ = test.evaluate(state, obs)
print(actions.requires_grad)
"""