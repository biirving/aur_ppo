from src.trainer.gymTrainer import gymTrainer
from src.utils.misc import ExpertTransition, normalizeTransition, store_returns
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from src.policies.sacBullet import sacBullet
import time, os, sys
from tqdm import tqdm

device = torch.device('cuda')


class awacTrainer(gymTrainer):
    # need a large batch size to learn stably
    def __init__(self, agent: sacBullet, anneal_lr=False, anneal_exp=False, total_time_steps=100000, num_env_steps=1024,
    num_processes=5, pretrain_episodes=5000, bc_episodes=100, num_eval_episodes=1, track=False, batch_size=64, expert_weight=0.01,
    save_path=None, run_id=0, transition_type='base'):
        super().__init__(total_time_steps, num_env_steps, num_processes, save_path, track, run_id, transition_type)
        self.agent = agent
        self.batch_size = batch_size
        self.anneal_lr = anneal_lr  
        self.anneal_exp = anneal_exp
        self.expert_weight = expert_weight
        self.num_updates = self.total_time_steps // self.batch_size
        self.num_eval_episodes = num_eval_episodes
        self.pretrain_episodes=pretrain_episodes
        self.track=track
        self.bc_episodes=bc_episodes

    def behavioral_clone(self, envs, agent, bc_episodes=100):
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
        
    def pretrain(self, pretrain_episodes):
        counter = 0
        update_counter = 0
        if pretrain_episodes > 0:
            planner_envs = self.envs
            planner_num_process = self.num_processes
            j = 0
            states, obs = planner_envs.reset()
            s = 0

            planner_bar = tqdm(total=pretrain_episodes)
            while j < pretrain_episodes:
                plan_actions = planner_envs.getNextAction()
                with torch.no_grad():
                    planner_actions_star_idx, planner_actions_star = self.agent.getActionFromPlan(plan_actions)
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

        pretrain_step = counter // self.batch_size
        pretrain_step = 0
        if pretrain_step > 0:
            for i in tqdm(range(pretrain_step)):
                batch = replay_buffer.sample(self.batch_size)
                agent.update(batch)

    def step_env(self, s, o, global_step):
        (u_a, a), lp, m, v = self.agent.act(s.cuda(), o.cuda(), deterministic=False)
        # expert action
        #true_action = self.envs.getNextAction()
        #u_e, e = self.agent.getActionFromPlan(true_action)
        self.envs.stepAsync(a)

        if len(self.replay_buffer) >= 100:
            batch = self.replay_buffer.sample(self.batch_size)
            self.agent.update(batch)

        n_s, n_o, r, d = self.envs.stepWait()
        for i, rew in enumerate(r):
            if rew != 0:
                print('reward', rew)
            self.returns.add_value(i, rew)

        done_idxes = torch.nonzero(d).squeeze(1)
        if done_idxes.shape[0] != 0:
            reset_states_, reset_obs_ = self.envs.reset_envs(done_idxes)
            for j, idx in enumerate(done_idxes):
                n_s[idx] = reset_states_[j]
                n_o[idx] = reset_obs_[j]
                discounted_return, episode_length = self.returns.calc_discounted_return(idx)
                self.writer.add_scalar("charts/discounted_episodic_return", discounted_return, global_step=global_step)
                self.writer.add_scalar("charts/episodic_length", episode_length, global_step=global_step)
        
        for i in range(self.num_processes):
            transition = ExpertTransitionPPO(s[i].numpy(), o[i].numpy(), u_a[i].numpy(), r[i].numpy(), d[i].numpy(), np.array(100), u_e[i].numpy(), lp[i].numpy(), v[i].numpy())
            self.replay_buffer.add(transition)

        s = copy.copy(n_s)
        o = copy.copy(o)

        return s, o

    def evaluate(self, global_step):
        s, o = self.eval_envs.reset()
        eval_ep = 0
        sum_r = 0

        eval_bar = tqdm(total=self.num_eval_episodes)
        while eval_ep < self.num_eval_episodes:
            u_a, a = self.agent.act(s.cuda(), o.cuda(), deterministic=True)
            s, o, r, d = self.eval_envs.step(a, auto_reset=True)

            for i, rew in enumerate(r):
                if rew != 0:
                    print('reward', rew)
                self.eval_returns.add_value(i, rew)
            eval_ep += d.sum().item()
            eval_bar.update(d.sum().item())

            done_idxes = torch.nonzero(d).squeeze(1)
            best_return = float('-inf')
            shortest_length = float('inf')
            if done_idxes.shape[0] != 0:
                reset_states_, reset_obs_ = self.envs.reset_envs(done_idxes)
                for j, idx in enumerate(done_idxes):
                    discounted_return, episode_length = self.eval_returns.calc_discounted_return(idx)
                    if discounted_return > best_return:
                        best_return = discounted_return
                        shortest_length = episode_length
                sum_r += best_return
        mean_r = sum_r / self.num_eval_episodes
        self.writer.add_scalar("charts/eval_discounted_episodic_return", mean_r, global_step=global_step)

    def run(self, simulator, env_config, planner_config, gym_id, actor, critic, encoder_type):
        self.initialize_env(simulator, env_config, planner_config, gym_id)
        self.agent.initNet(actor, critic, encoder_type)
        if self.track:
            import wandb
            wandb.init(project='sac', entity='Aurelian', sync_tensorboard=True, config=None, name='ppo_' + gym_id)
        self.writer = SummaryWriter(f"runs/{gym_id}")
        self.set_threads_and_seeds(1)

        #mse_envs = self.envs
        #self.behavioral_clone(mse_envs, self.agent, self.bc_episodes)

        if self.do_pretraining:
            print('Pretraining...')
            self.pretrain(self.pretrain_episodes)
        self.evaluate(0)

        start_time = time.time()
        n_s, n_o = self.envs.reset()
        for t in tqdm(range(total_steps)):
            n_s, n_o = self.step_env(n_s, n_o, t)
        
        self.save_agent(gym_id, self.save_path)
        self.envs.close()
        self.test_envs.close()
        self.writer.close()