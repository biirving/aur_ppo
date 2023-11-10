import torch
from ppo import ppo
import os, sys, argparse, time

self.gym_id = params['gym_id']
		self.exp_name = params['exp_name']
		self.seed = params['seed']
		self.num_steps = params['num_steps']
		self.gae = params['gae']
		self.total_timesteps = params['total_timesteps']
		self.batch_size = params['batch_size']
		self.num_updates = params['num_updates']
		self.anneal_lr = params['anneal_lr']
		self.gae_lambda = params['gae_lambda']
		self.num_update_epochs = params['num_update_epochs']
		self.num_envs = params['num_envs']
		self.num_minibatches = params['num_minibatches']
		self.batch_size = self.num_steps * self.num_envs
		self.minibatch_size = int(self.batch_size // self.num_minibatches)
		self.entropy_coeff = params['entropy_coeff']
		self.value_coeff = params['value_coeff']
		self.grad_norm = params['grad_norm']
		self.target_kl = params['target_kl']
		self.agent = actor_critic()
		self.norm_adv = params['norm_adv']



if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-id', '--gym_id', help='Id of the environment that we will use', default='CartPole-v1')
	parser.add_argument('-s', '--seed', help='Seed for experiment', default=1.0)
	parser.add_argument('-ns', '--num_steps', help='Number of steps that the environment should take', default=400)
	parser.add_argument('-gae', '--gae', help='Generalized Advantage Estimation flag', default=True)
	parser.add_argument('-t', '--total_timesteps', help='Total number of timesteps that we will take', default=25000)
	parser.add_argument('-')
	args = parser.parse_args()

	params = {
		'gym_id':args.gym_id,

	}