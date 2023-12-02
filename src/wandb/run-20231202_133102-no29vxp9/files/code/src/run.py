#!/usr/bin/env python
import torch
from ppo import ppo
import os, sys, argparse, time
import matplotlib.pyplot as plt
import numpy as np

def plot_curves(arr_list, legend_list, x_indices, color_list, ylabel, fig_title):
	plt.clf()
	fig, ax = plt.subplots(figsize=(12, 8))
	ax.set_ylabel(ylabel)
	ax.set_xlabel("Steps")
	h_list = []

	for arr, legend, color in zip(arr_list, legend_list, color_list):
		arr_err = arr.std(axis=0) / np.sqrt(arr.shape[0])
		h = ax.plot(x_indices, arr.mean(axis=0), color=color, label=legend)
		arr_err = 1.96 * arr_err
		ax.fill_between(x_indices, arr.mean(axis=0) - arr_err, arr.mean(axis=0) + arr_err, alpha=0.3,
		                color=color)
		h_list.append(h)
	ax.set_title(f"{fig_title}")
	#ax.legend(handles=h_list)
	plt.savefig('total_returns.png')
	#plt.show()

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-id', '--gym_id', type=str, help='Id of the environment that we will use', default='CartPole-v1')
	parser.add_argument('-s', '--seed', type=float, help='Seed for experiment', default=1.0)
	parser.add_argument('-ns', '--num_steps', type=int, help='Number of steps that the environment should take', default=128)
	parser.add_argument('-gae', '--gae', type=bool, help='Generalized Advantage Estimation flag', default=True)
	parser.add_argument('-t', '--total_timesteps', type=int, help='Total number of timesteps that we will take', default=500000)
	parser.add_argument('-al', '--anneal_lr', type=bool, help='How to anneal our learning rate', default=True)
	parser.add_argument('-gl', '--gae_lambda', type=float, help="the lambda for the general advantage estimation", default=0.95)
	parser.add_argument('-ue', '--num_update_epochs', type=int, help='The  number of update epochs for the policy', default=4)
	parser.add_argument('-ne', '--num_envs', type=int, help='Number of environments to run in our vectorized setup', default=4)
	parser.add_argument('-nm', '--num_minibatches', type=int, help='Number of minibatches', default=4)
	parser.add_argument('-ec', '--entropy_coeff', type=float, help='Coefficient for entropy', default=0.001)
	parser.add_argument('-vf', '--value_coeff', type=float, help='Coefficient for values', default=0.5)
	parser.add_argument('-cf', '--clip_coeff', type=float, help="the surrogate clipping coefficient",  default=0.2)
	parser.add_argument('-cvl', '--clip_vloss', type=bool, help="Clip the value loss", default=True)
	parser.add_argument('-mgn', '--max_grad_norm', type=float, help='the maximum norm for the gradient clipping', default=0.5)
	parser.add_argument('-tkl', '--target_kl',type=float, help='The KL divergence that we will not exceed', default=0.2)
	parser.add_argument('-na', '--norm_adv', type=bool, help='Normalize advantage estimates', default=True)
	parser.add_argument('-p', '--capture_video', type=bool, help='Whether to capture the video or not', default=False)
	parser.add_argument('-d', '--hidden_dim', type=int, help='Hidden dimension of the neural networks in the actor critic', default=64)
	parser.add_argument('-c', '--continuous', type=bool, help='Is the action space continuous',default=False)
	parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate for our agent', default=2.5e-4)
	parser.add_argument('-exp', '--exp_name', type=str, help='Experiment name', default='CartPole PPO')
	parser.add_argument('-nl', '--num_layers', type=int, help='The number of layers in our actor and critic', default=2)
	parser.add_argument('-do', '--dropout', type=float, help='Dropout in our actor and critic', default=0.0)
	parser.add_argument('-g', '--gamma', type=float, help='Discount value for rewards', default=0.99)
	parser.add_argument('-tr', '--track', type=bool, help='Track the performance of the environment', default=True)
	parser.add_argument('-tri', '--trials', type=int, help='Number of trials to run', default=1)
	args = parser.parse_args()

	params = {
		'gym_id':args.gym_id,
		'seed':args.seed,
		'num_steps':args.num_steps,
		'gae':args.gae,
		'total_timesteps':args.total_timesteps,
		'anneal_lr':args.anneal_lr,
		'gae_lambda':args.gae_lambda,
		'num_update_epochs':args.num_update_epochs,
		'num_envs':args.num_envs,
		'num_minibatches':args.num_minibatches,
		'entropy_coeff':args.entropy_coeff,
		'value_coeff':args.value_coeff,
		'clip_coeff':args.clip_coeff,
		'clip_vloss':args.clip_vloss,
		'max_grad_norm':args.max_grad_norm,
		'target_kl':args.target_kl,
		'norm_adv':args.norm_adv,
		'capture_video':args.capture_video,
		'hidden_dim':args.hidden_dim,
		'continuous':args.continuous,
		'learning_rate':args.learning_rate, 
		'hidden_dim':args.hidden_dim,
		'exp_name':args.exp_name,
		'num_layers':args.num_layers,
		'dropout':args.dropout,
		'gamma':args.gamma,
		'track':args.track
	}

	#all_returns = []
	#all_episode_lengths = []
	#trials = args.trials
	#for _ in range(trials):
	to_run = ppo(params)
	total_returns, total_episode_lengths, x_indices = to_run.train()
	#all_returns.append(total_returns)
	#all_episode_lengths.append(total_episode_lengths)

	# Find the minimum length
	#min_length = min(len(sublist) for sublist in all_returns)
	#Trim each sublist to the minimum length
	#trim_returns = [sublist[:min_length] for sublist in all_returns]
	# then trim all of the trials down to the minimum length
	#plot_curves(np.array([trim_returns]), ['Agent'], x_indices[:min_length], ['red'], 'Total Return', args.gym_id)
	# plot with confidence bands and stuff