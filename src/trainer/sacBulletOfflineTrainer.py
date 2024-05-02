from src.trainer.sacBulletTrainer import sacBulletTrainer
from src.utils.misc import ExpertTransitionOffline, normalizeTransition, store_returns, normalizeTransitionOffline
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from src.policies.sacBullet import sacBullet
import time, os, sys
from tqdm import tqdm
import copy

device = torch.device('cuda')

class sacBulletOfflineTrainer(sacBulletTrainer):
    def __init__(self, agent: sacBullet, anneal_lr=False, anneal_exp=False, total_time_steps=100000, num_env_steps=1024,
    num_processes=5, pretrain_episodes=5000, num_eval_episodes=1000, track=False, 
    batch_size=64, save_path=None, training_episodes=100000, bc_episodes=100, expert_weight=1e-3,
    episode_save_path=None, run_id=0, mixed_expert_episodes=10000, random_episodes=5000, random_action_frequency=10, expert_agents=None):
        super().__init__(agent, anneal_lr, anneal_exp, total_time_steps,
            num_env_steps, num_processes, pretrain_episodes, bc_episodes, num_eval_episodes, track, batch_size, expert_weight, save_path, run_id)
        self.training_episodes=training_episodes
        self.episode_save_path=episode_save_path
        self.mixed_expert_episodes=mixed_expert_episodes
        self.random_episodes=random_episodes
        self.random_action_frequency=random_action_frequency

    def collect_mixed_expert_episodes(self, counter):
        # should allow for different agent inputs to collect the rollouts
        training_agent=self.agent
        if self.mixed_expert_episodes > 0:
            training_envs = self.envs
            training_num_processes = self.num_processes
            j = 0
            states, obs = training_envs.reset()
            s = 0
            training_bar = tqdm(total=self.mixed_expert_episodes)
            # should have some bad episodes
            
            # every 10th episode should be a random policy
            while j < self.mixed_expert_episodes:
                train_actions = training_envs.getNextAction()
                with torch.no_grad():
                    training_actions_star_idx, training_actions_star = training_agent.getActionFromPlan(train_actions)
                    random_actions_star_idx, random_actions_star =  training_agent.act(states.to(device), obs.to(device), deterministic=False)
                    if counter % self.random_action_frequency == 0:
                        states_, obs_, rewards, dones = training_envs.step(random_actions_star, auto_reset=True)
                    else:
                        states_, obs_, rewards, dones = training_envs.step(training_actions_star, auto_reset=True)
                for i in range(training_num_processes):
                    transition = ExpertTransitionOffline(states[i].numpy(), obs[i].numpy(), random_actions_star_idx[i].numpy(),
                                                rewards[i].numpy(), states_[i].numpy(), obs_[i].numpy(), 
                                                dones[i], np.array(100), np.array(1), training_actions_star_idx[i].numpy())
                    transition = normalizeTransitionOffline(transition)
                    self.replay_buffer.add(transition)
                    counter += 1
                states = copy.copy(states_)
                obs = copy.copy(obs_)
                j += dones.sum().item()
                s += rewards.sum().item()
                training_bar.set_description('{:.3f}/{}, AVG: {:.3f}'.format(s, j, float(s) / j if j != 0 else 0))
                training_bar.update(dones.sum().item())

        return counter

    def collect_random_episodes(self, counter):
        training_agent=self.agent
        if self.random_episodes > 0:
            training_envs = self.envs
            training_num_processes = self.num_processes
            j = 0
            states, obs = training_envs.reset()
            s = 0
            training_bar = tqdm(total=self.random_episodes)
            # should have some bad episodes
            
            # every 10th episode should be a random policy
            while j < self.random_episodes:
                train_actions = training_envs.getNextAction()
                with torch.no_grad():
                    training_actions_star_idx, training_actions_star = training_agent.getActionFromPlan(train_actions)
                    random_actions_star_idx, random_actions_star =  training_agent.act(states.to(device), obs.to(device), deterministic=False)
                    states_, obs_, rewards, dones = training_envs.step(random_actions_star, auto_reset=True)

                for i in range(training_num_processes):
                    transition = ExpertTransitionOffline(states[i].numpy(), obs[i].numpy(), random_actions_star_idx[i].numpy(),
                                                rewards[i].numpy(), states_[i].numpy(), obs_[i].numpy(), 
                                                dones[i], np.array(100), np.array(1), training_actions_star_idx[i].numpy())

                    transition = normalizeTransitionOffline(transition)
                    self.replay_buffer.add(transition)
                    counter += 1
                states = copy.copy(states_)
                obs = copy.copy(obs_)
                j += dones.sum().item()
                s += rewards.sum().item()
                training_bar.set_description('{:.3f}/{}, AVG: {:.3f}'.format(s, j, float(s) / j if j != 0 else 0))
                training_bar.update(dones.sum().item())

        return counter

    def run(self, simulator, env_config, planner_config, gym_id, actor, critic, encoder_type):
        counter = 0
        self.initialize_env(simulator, env_config, planner_config, gym_id)
        self.agent.initNet(actor, critic, encoder_type)
        if self.track:
            import wandb
            wandb.init(project='sac', entity='Aurelian', sync_tensorboard=True, config=None, name='ppo_' + gym_id)
        self.writer = SummaryWriter(f"runs/{gym_id}")
        self.set_threads_and_seeds(1)

        mse_envs = self.envs
        self.behavioral_clone(mse_envs, self.agent, self.bc_episodes)
        self.evaluate(0)
        counter = self.collect_mixed_expert_episodes(counter)
        counter = self.collect_random_episodes(counter)
        #counter = self.collect_input_expert_episodes(counter)

        if self.episode_save_path is not None:
            self.replay_buffer.save_buffer(self.episode_save_path, self.run_id)

        offline_updates = counter // self.batch_size
        assert offline_updates > 0
        for i in tqdm(range(offline_updates)):
            batch = self.replay_buffer.sample(self.batch_size)
            self.agent.update(batch)
            # evaluate every 100 updates
            if i % 100 == 0:
                self.evaluate(i)

        self.agent.save_agent(gym_id, self.save_path)
        self.envs.close()
        self.eval_envs.close()
        self.writer.close()
