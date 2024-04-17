from src.trainer.bulletTrainer import bulletTrainer
from src.utils.misc import ExpertTransitionPPO, normalizeTransition, store_returns
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from src.policies.ppoBullet import ppoBullet
import time, os, sys
from tqdm import tqdm

device = torch.device('cuda')

class ppoBulletTrainer(bulletTrainer):
    def __init__(self, agent: ppoBullet, anneal_lr=False, anneal_exp=False,total_time_steps=10000, num_env_steps=128, num_processes=5, num_eval_episodes=100):
        super().__init__(total_time_steps, num_env_steps, num_processes)
        self.agent = agent
        self.anneal_lr = anneal_lr  
        self.anneal_exp = anneal_exp
        self.expert_weight = 0.01
        self.ppo_batch = self.num_processes * self.num_env_steps
        self.num_updates = self.total_time_steps // self.ppo_batch
        self.num_eval_episodes = num_eval_episodes
		

    def pretrain(self):
        states, obs = self.envs.reset()
        pretrain_episodes = 100
        pretrain_batch_size = 16
        expert_actions = []
        agent_actions = []
        j = 0
        while j < pretrain_episodes:
            with torch.no_grad():
                true_action = self.envs.getNextAction()
                unscaled, scaled = self.agent.getActionFromPlan(true_action)
                expert_actions.append(unscaled.cpu().numpy())
                (scaled_agent, unscaled_agent), logprob, _, value = self.agent.act(states.to(device), obs.to(device))
                agent_actions.append(unscaled_agent.cpu().numpy())
            states, obs, reward, dones = self.envs.step(scaled, auto_reset=True)
            j += dones.sum().item()

        expert_tensor = torch.tensor(np.stack([a for a in expert_actions])).squeeze(dim=0)
        agent_tensor = torch.tensor(np.stack([a for a in agent_actions])).squeeze(dim=0)

        total_pretrain_steps = expert_tensor.shape[0]
        b_inds = np.arange(total_pretrain_steps)

        for index in tqdm(range(0, total_pretrain_steps, pretrain_batch_size)):
            mb_inds = b_inds[index:index+pretrain_batch_size]
            expert_loss = nn.functional.mse_loss(agent_tensor[mb_inds].cuda().requires_grad_(True), expert_tensor[mb_inds].cuda())
            self.agent.pretrain_update(expert_loss)
            print('expert loss', expert_loss)

    def step_env(self, s, o, global_step):
        (u_a, a), lp, m, v = self.agent.act(s.cuda(), o.cuda())
        n_s, n_o, r, d = self.envs.step(a)
        for i, rew in enumerate(r):
            if rew != 0:
                print('reward', rew)
            self.returns.add_value(i, rew)

        done_idxes = torch.nonzero(d).squeeze(1)
        if done_idxes.shape[0] != 0:
            reset_states_, reset_obs_ = self.envs.reset_envs(done_idxes)
            for j, idx in enumerate(done_idxes):
                discounted_return, episode_length = self.returns.calc_discounted_return(idx)
                self.writer.add_scalar("charts/discounted_episodic_return", discounted_return, global_step=global_step)
                self.writer.add_scalar("charts/episodic_length", episode_length, global_step=global_step)
        
        for i in range(self.num_processes):
            transition = ExpertTransitionPPO(s[i].numpy(), o[i].numpy(), u_a[i].numpy(), r[i].numpy(), d[i].numpy(), np.array(100), 0, lp[i].numpy(), v.numpy())
            self.replay_buffer.add(transition)

        return n_s, n_o, d

    def evaluate(self, global_step):
        eval_bar = tqdm(total=self.num_eval_episodes)
        s, o = self.eval_envs.reset()
        eval_ep = 0
        sum_r = 0
        while eval_ep < self.num_eval_episodes:
            (u_a, a), _, _, _ = self.agent.act(s.cuda(), o.cuda(), deterministic=True)
            s, o, r, d = self.eval_envs.step(a, auto_reset=True)

            for i, rew in enumerate(r):
                if rew != 0:
                    print('reward', rew)
                self.eval_returns.add_value(i, rew)
            eval_ep += d.sum().item()

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

    def run(self, simulator, env_config, planner_config, gym_id, actor, critic):
        self.initialize_env(simulator, env_config, planner_config, gym_id)
        self.agent.initNet(actor, critic)
        if self.track:
            import wandb
            wandb.init(project='ppo', entity='Aurelian', sync_tensorboard=True, config=None, name=gym_id + '_' + str(self.learning_rate) + '_' + 
                       str(self.value_coeff) + '_' + str(self.entropy_coeff) + '_' + str(self.clip_coeff) + '_' + str(self.num_minibatches))
        self.writer = SummaryWriter(f"runs/{gym_id}")
        self.set_threads_and_seeds(1)

        if self.do_pretraining:
            self.pretrain()

        start_time = time.time()
        self.global_step = 0
        next_state, next_obs = self.envs.reset()
        next_done = torch.zeros(self.num_processes).to(device)

        for update in tqdm(range(1, self.num_updates + 1)):
            t0 = time.time()
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / self.num_updates
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            if self.anneal_exp:
                frac = 1 - ((update - 1)/self.num_updates)
                self.expert_weight *= frac

            for step in tqdm(range(0, self.num_env_steps)):
                self.global_step += 1 * self.num_processes 
                n_s, n_o, n_d = self.step_env(next_state, next_obs, self.global_step)    
                batch = self.replay_buffer.sample(self.replay_buffer.__len__())
                #with torch.no_grad():
                n_s = n_s.reshape(n_s.size(0), 1, 1, 1).repeat(1, 1, n_o.shape[2], n_o.shape[3])
                n_o  = torch.cat([n_o, n_s], dim=1)
                self.agent.update(batch, n_o.cuda(), n_d.cuda())
                self.replay_buffer.clear()

                if self.global_step % 1000 == 0:
                    self.evaluate(self.global_step)

                self.returns.reset()
