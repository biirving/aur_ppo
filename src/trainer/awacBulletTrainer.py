from src.trainer.sacBulletTrainer import sacBulletTrainer
from src.utils.misc import ExpertTransitionOffline, normalizeTransition, store_returns, normalizeTransitionOffline
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from src.policies.sacBullet import sacBullet
from src.policies.awacBullet import awacBullet 
import time, os, sys
from tqdm import tqdm
import copy

device = torch.device('cuda')

class awacBulletTrainer(sacBulletTrainer):
    def __init__(self, agent: awacBullet, anneal_lr=False, anneal_exp=False,
    total_time_steps=100000, num_env_steps=1024,
    num_processes=5, pretrain_episodes=5000, num_eval_episodes=10, track=False, 
    batch_size=128, save_path=None, training_episodes=100000, bc_episodes=200, expert_weight=1e-3,
    episode_save_path=None, run_id=0, mixed_expert_episodes=1000, random_episodes=500, random_action_frequency=10, expert_agents=None, 
    load_episodes=False, buffer_path=None, buffer_run_id=None, transition_type='base'):
        super().__init__(agent, anneal_lr, anneal_exp, total_time_steps,
            num_env_steps, num_processes, pretrain_episodes, bc_episodes, num_eval_episodes, track, batch_size, expert_weight, save_path, run_id, transition_type)
        self.training_episodes=training_episodes
        self.episode_save_path=episode_save_path
        self.mixed_expert_episodes=mixed_expert_episodes
        self.random_episodes=random_episodes
        self.random_action_frequency=random_action_frequency
        self.load_episodes=load_episodes
        self.buffer_path=buffer_path
        self.buffer_run_id=buffer_run_id

    def behavioral_clone_collect(self, envs, agent, bc_episodes=100):
        """
        Behavior clone function, but we collect the rollouts. 
        """
        print('Collecting behavioral cloning episodes...')
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
                random_actions_star_idx, random_actions_star =  agent.act(states.to(device), obs.to(device), deterministic=False)
                expert_actions.append(unscaled.cpu().numpy())
                obs_to_add = torch.cat([obs, states.reshape(states.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])], dim=1)
            obs_list.append(obs_to_add.cpu().numpy())
            states_, obs_, rewards, dones = envs.step(scaled, auto_reset=True)

            for i in range(self.num_processes):
                transition = ExpertTransitionOffline(states[i].numpy(), obs[i].numpy(), random_actions_star_idx[i].numpy(),
                                        rewards[i].numpy(), states_[i].numpy(), obs_[i].numpy(), 
                                        dones[i], np.array(100), np.array(1), unscaled[i].numpy())
                transition = normalizeTransitionOffline(transition)
                self.replay_buffer.add(transition)

            states = copy.copy(states_)
            obs = copy.copy(obs_)
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


    def collect_mixed_expert_episodes(self):
        print('Collecting mixed expert episodes...')
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
                    if self.replay_buffer.__len__() % self.random_action_frequency == 0:
                        states_, obs_, rewards, dones = training_envs.step(random_actions_star, auto_reset=True)
                    else:
                        states_, obs_, rewards, dones = training_envs.step(training_actions_star, auto_reset=True)
                for i in range(training_num_processes):
                    transition = ExpertTransitionOffline(states[i].numpy(), obs[i].numpy(), random_actions_star_idx[i].numpy(),
                                                rewards[i].numpy(), states_[i].numpy(), obs_[i].numpy(), 
                                                dones[i], np.array(100), np.array(1), training_actions_star_idx[i].numpy())
                    transition = normalizeTransitionOffline(transition)
                    self.replay_buffer.add(transition)
                states = copy.copy(states_)
                obs = copy.copy(obs_)
                j += dones.sum().item()
                s += rewards.sum().item()
                training_bar.set_description('{:.3f}/{}, AVG: {:.3f}'.format(s, j, float(s) / j if j != 0 else 0))
                training_bar.update(dones.sum().item())


    def collect_random_episodes(self):
        print('collecting random episodes...')
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
                states = copy.copy(states_)
                obs = copy.copy(obs_)
                j += dones.sum().item()
                s += rewards.sum().item()
                training_bar.set_description('{:.3f}/{}, AVG: {:.3f}'.format(s, j, float(s) / j if j != 0 else 0))
                training_bar.update(dones.sum().item())

    # should we use some annealing structure?
    # 
    def collect_input_expert_episodes(self, experts):
        training_agent=self.agent

        # we collect these episodes from a mixture of policies
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
                states = copy.copy(states_)
                obs = copy.copy(obs_)
                j += dones.sum().item()
                s += rewards.sum().item()
                training_bar.set_description('{:.3f}/{}, AVG: {:.3f}'.format(s, j, float(s) / j if j != 0 else 0))
                training_bar.update(dones.sum().item())

    def run(self, simulator, env_config, planner_config, gym_id, actor, critic, encoder_type):
        self.initialize_env(simulator, env_config, planner_config, gym_id)
        self.agent.initNet(actor, critic, encoder_type)
        if self.track:
            import wandb
            wandb.init(project='sac', entity='Aurelian', sync_tensorboard=True, config=None, name='ppo_' + gym_id)
        self.writer = SummaryWriter(f"runs/{gym_id}")
        self.set_threads_and_seeds(1)

        mse_envs = self.envs

        if self.load_episodes:
            print('Loading buffer...')
            self.replay_buffer.load_buffer(self.buffer_path, gym_id, self.buffer_run_id)
        else:
            print('Starting rollout...')
            self.collect_mixed_expert_episodes()
            self.behavioral_clone_collect(mse_envs, self.agent, self.bc_episodes)
            self.collect_random_episodes()

            # we should have partial loading + saving depending on what the buffer contains
            if self.episode_save_path is not None:
                print('Saving episodes...')
                self.replay_buffer.save_buffer(self.episode_save_path, gym_id, self.buffer_run_id)

        offline_updates = self.replay_buffer.__len__() // self.batch_size

        assert offline_updates > 0
        for i in tqdm(range(offline_updates)):
            batch = self.replay_buffer.sample(self.batch_size)
            self.agent.update(batch)
            # evaluate every 100 updates
            if i % 1 == 0:
                self.evaluate(i)


        self.agent.save_agent(gym_id, self.save_path,)
        self.envs.close()
        self.eval_envs.close()
        self.writer.close()
