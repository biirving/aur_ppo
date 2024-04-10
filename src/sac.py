# sanity check
# open ai SAC implementation, with a few changes

from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import time
import sys

sys.path.append('/work/nlp/b.irving/aur_ppo/src')
import models.sac_core as core
from utils.str2bool import str2bool
from env_wrapper_2 import EnvWrapper

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import collections
import copy

sys.path.append('/home/benjamin/Desktop/ml/BulletArm/bulletarm_baselines')
from bulletarm_baselines.equi_rl.agents.sac import SAC
from bulletarm_baselines.equi_rl.networks.sac_net import SACCritic, SACGaussianPolicy
from bulletarm_baselines.equi_rl.networks.equivariant_sac_net import EquivariantSACActor, EquivariantSACCritic, EquivariantSACActorDihedral, EquivariantSACCriticDihedral

ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done step_left expert')

import numpy.random as npr
class QLearningBuffer:
    def __init__(self, size):
        self._storage = []
        self._max_size = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def __getitem__(self, key):
        return self._storage[key]

    def __setitem__(self, key, value):
        self._storage[key] = value

    def add(self, data):
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._max_size

    def sample(self, batch_size):
        batch_indexes = npr.choice(self.__len__(), batch_size).tolist()
        batch = [self._storage[idx] for idx in batch_indexes]
        return batch

    def getSaveState(self):
        return {
            'storage': self._storage,
            'max_size': self._max_size,
            'next_idx': self._next_idx
        }

    def loadFromState(self, save_state):
        self._storage = save_state['storage']
        self._max_size = save_state['max_size']
        self._next_idx = save_state['next_idx']

def _loadBatchToDevice(batch, device=torch.device('cuda')):
        """
        Load batch into pytorch tensor
        :param batch: list of transitions
        :return: states_tensor, obs_tensor, action_tensor, rewards_tensor, next_states_tensor, next_obs_tensor,
                 non_final_masks, step_lefts_tensor, is_experts_tensor
        """
        states = []
        images = []
        xys = []
        rewards = []
        next_states = []
        next_obs = []
        dones = []
        step_lefts = []
        is_experts = []
        for d in batch:
            states.append(d.state)
            images.append(d.obs)
            xys.append(d.action)
            rewards.append(d.reward.squeeze())
            next_states.append(d.next_state)
            next_obs.append(d.next_obs)
            dones.append(d.done)
            step_lefts.append(d.step_left)
            is_experts.append(d.expert)
        states_tensor = torch.tensor(np.stack(states)).long().to(device)
        obs_tensor = torch.tensor(np.stack(images)).to(device)
        if len(obs_tensor.shape) == 3:
            obs_tensor = obs_tensor.unsqueeze(1)
        action_tensor = torch.tensor(np.stack(xys)).to(device)
        rewards_tensor = torch.tensor(np.stack(rewards)).to(device)
        next_states_tensor = torch.tensor(np.stack(next_states)).long().to(device)
        next_obs_tensor = torch.tensor(np.stack(next_obs)).to(device)
        if len(next_obs_tensor.shape) == 3:
            next_obs_tensor = next_obs_tensor.unsqueeze(1)
        dones_tensor = torch.tensor(np.stack(dones)).int()
        non_final_masks = (dones_tensor ^ 1).float().to(device)
        step_lefts_tensor = torch.tensor(np.stack(step_lefts)).to(device)
        is_experts_tensor = torch.tensor(np.stack(is_experts)).bool().to(device)

        # scale observation from int to float
        obs_tensor = obs_tensor/255*0.4
        next_obs_tensor = next_obs_tensor/255*0.4

        return states_tensor, obs_tensor, action_tensor, rewards_tensor, next_states_tensor, \
               next_obs_tensor, non_final_masks, step_lefts_tensor, is_experts_tensor

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.

    What are other experience replay buffers that could be used instead?
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.states_buf_1 = np.zeros(core.combined_shape(size, 1))
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.states_buf_2 = np.zeros(core.combined_shape(size, 1))
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    # how are the states of the grippers going to be stored?
    def store(self, obs, state, act, rew, next_obs, next_state, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.states_buf_1[self.ptr] = state
        self.states_buf_2[self.ptr] = next_state
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.ptr, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     states=self.states_buf_1[idxs],
                     states2=self.states_buf_2[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
    
# so we are normalizing the transitions
#def normalizeTransition(obs, next_obs):
#    obs = np.clip(obs, 0, 0.32)
#    obs = obs/0.4*255
#    obs = obs.astype(np.uint8)

#    next_obs = np.clip(next_obs, 0, 0.32)
#    next_obs = next_obs/0.4*255
#    next_obs = next_obs.astype(np.uint8)
#    return obs, next_obs

def normalizeTransition(d: ExpertTransition):
    obs = np.clip(d.obs, 0, 0.32)
    obs = obs/0.4*255
    obs = obs.astype(np.uint8)

    next_obs = np.clip(d.next_obs, 0, 0.32)
    next_obs = next_obs/0.4*255
    next_obs = next_obs.astype(np.uint8)

    return ExpertTransition(d.state, obs, d.action, d.reward, d.next_state, next_obs, d.done, d.step_left, d.expert)

# moving decode and get action from plan functions

dpos = 0.05
dr=np.pi/8
n_a=5
p_range = torch.tensor([0, 1])
dtheta_range = torch.tensor([-dr, dr])
dx_range = torch.tensor([-dpos, dpos])
dy_range = torch.tensor([-dpos, dpos])
dz_range = torch.tensor([-dpos, dpos])	
n_a = n_a

def decodeActions(*args):
    unscaled_p = args[0]
    unscaled_dx = args[1]
    unscaled_dy = args[2]
    unscaled_dz = args[3]

    p = 0.5 * (unscaled_p + 1) * (p_range[1] - p_range[0]) + p_range[0]
    dx = 0.5 * (unscaled_dx + 1) * (dx_range[1] - dx_range[0]) + dx_range[0]
    dy = 0.5 * (unscaled_dy + 1) * (dy_range[1] - dy_range[0]) + dy_range[0]
    dz = 0.5 * (unscaled_dz + 1) * (dz_range[1] - dz_range[0]) + dz_range[0]

    if n_a == 5:
        unscaled_dtheta = args[4]
        dtheta = 0.5 * (unscaled_dtheta + 1) * (dtheta_range[1] - dtheta_range[0]) + dtheta_range[0]
        actions = torch.stack([p, dx, dy, dz, dtheta], dim=1)
        unscaled_actions = torch.stack([unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz, unscaled_dtheta], dim=1)
    else:
        actions = torch.stack([p, dx, dy, dz], dim=1)
        unscaled_actions = torch.stack([unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz], dim=1)
    return unscaled_actions, actions

# scaled actions
def getActionFromPlan(plan):
    def getUnscaledAction(action, action_range):
        unscaled_action = 2 * (action - action_range[0]) / (action_range[1] - action_range[0]) - 1
        return unscaled_action
    dx = plan[:, 1].clamp(*dx_range)
    p = plan[:, 0].clamp(*p_range)
    dy = plan[:, 2].clamp(*dy_range)
    dz = plan[:, 3].clamp(*dz_range)
    unscaled_p = getUnscaledAction(p, p_range)
    unscaled_dx = getUnscaledAction(dx, dx_range)
    unscaled_dy = getUnscaledAction(dy, dy_range)
    unscaled_dz = getUnscaledAction(dz, dz_range)
    if n_a == 5:
        dtheta = plan[:, 4].clamp(*dtheta_range)
        unscaled_dtheta = getUnscaledAction(dtheta, dtheta_range)
        return decodeActions(unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz, unscaled_dtheta)
    else:
        return decodeActions(unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz)

def sac(envs, test_envs, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        num_processes=1, steps_per_epoch=1000, epochs=4, replay_size=int(1e5), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=64, start_steps=10000, 
        update_after=1000, update_every=50, pretrain_episodes=50, num_test_episodes=10, 
        max_ep_len=100,track=False, save_freq=1, gym_id=None, device=torch.device('cpu')):
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """


    if track:
        import wandb
        wandb.init(project='ppo',entity='Aurelian',sync_tensorboard=True,config=None,name=gym_id + '_' + str(lr)) #+ '_' 
       # str(self.value_coeff) + '_' + str(self.entropy_coeff) + '_' + str(self.clip_coeff) + '_' + str(self.num_minibatches),monitor_gym=True,save_code=True)
    writer = SummaryWriter(f"runs/{gym_id}")
    writer.add_text("parameters/what", "what")
    #writer.add_text(
    #"hyperparameters",
    #"|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{str(self.params_dict[key])}|" for key in self.params_dict])),
    #)


    torch.manual_seed(seed)
    np.random.seed(seed)

    # will this make two copies
    env, test_env = envs, test_envs
    obs_dim = (1, 128, 128)
    #act_dim = env.action_space.shape[0]
    act_dim=5

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    #act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks

    ac = core.MLPActorCritic().to(device)
    ac_targ = deepcopy(ac.to(device))

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    #for p in ac_targ.parameters():
    #    p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    #q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
    q_params = itertools.chain(ac.critic.parameters())

    # Experience buffer

    #replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    replay_buffer = QLearningBuffer(replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.critic])

    # Set up function for computing SAC Q-losses
    def compute_loss_q(s, o, a, r, s2, o2, d):
        
        

        # Bellman backup for Q functions
        with torch.no_grad():
            s_tile = s.reshape(s.size(0), 1, 1, 1).repeat(1, 1, o.shape[2], o.shape[3])
            cat_o = torch.cat([o, s_tile], dim=1).to(device)

            s_tile_2 = s2.reshape(s2.size(0), 1, 1, 1).repeat(1, 1, o2.shape[2], o2.shape[3])
            cat_o_2 = torch.cat([o2, s_tile_2], dim=1).to(device)
            # Target actions come from *current* policy
            a_coded, logp_a2, mean = ac.sample(cat_o_2)

            #u_a2, a2 = decodeActions(*[a_coded[:, i] for i in range(n_a)])

            # Target Q-values
            #q1_pi_targ = ac_targ.q1(cat_o_2, u_a2)
            #q2_pi_targ = ac_targ.q2(cat_o_2, u_a2)
            q1_pi_targ, q2_pi_targ = ac_targ.critic(cat_o_2, a_coded)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ).to(device)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        #q1 = ac.q1(cat_o,a)
        #q2 = ac.q2(cat_o,a)
        q1, q2 = ac.critic(cat_o, a)
        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                      Q2Vals=q2.cpu().detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(s, o):

        # that means we have to compute the cat obs manually
        # convert to tensors before processing
        with torch.no_grad():
            s_tile = s.reshape(s.size(0), 1, 1, 1).repeat(1, 1, o.shape[2], o.shape[3])
            cat_o = torch.cat([o, s_tile], dim=1).to(device)

        # need to change this
        a_coded, logp_pi, mean = ac.sample(cat_o)
        #u_pi, pi = decodeActions(*[a_coded[:, i] for i in range(n_a)])

        #q1_pi = ac.q1(cat_o, pi)
        #q2_pi = ac.q2(cat_o, pi)
        q1_pi, q2_pi = ac.critic(cat_o, a_coded)

        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.cpu().detach().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(ac.critic.parameters(), lr=lr)

    # Set up model saving

    def update(data, num_update):
        # First run one gradient descent step for Q1 and Q2

        # should load the data in here

        s, o, a, r, s2, o2, d, _, _ = _loadBatchToDevice(data)

        loss_q, q_info = compute_loss_q(s, o, a, r, s2, o2, d)

        q_optimizer.zero_grad()
        loss_q.backward()
        q_optimizer.step()

        # Record things

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        #for p in q_params:
        #    p.requires_grad = False

        # Next run one gradient descent step for pi.
        loss_pi, pi_info = compute_loss_pi(s, o)

        pi_optimizer.zero_grad()
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        #for p in q_params:
        #    p.requires_grad = True

        # Record things

        #with torch.no_grad():
        
        tau = 1e-2
        if num_update % 100 == 0:
            for t_param, l_param in zip(
                    ac_targ.critic.parameters(), ac.critic.parameters()
            ):
                t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        """
        for t_param, l_param in zip(
                ac_targ.q1.parameters(), ac.q1.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)
        for t_param, l_param in zip(
                ac_targ.q2.parameters(), ac.q2.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)
        """
            #for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                #p_targ.data.mul_(polyak)
                #p_targ.data.add_((1 - polyak) * p.data)

    def get_action(s, o, deterministic=False):
        with torch.no_grad():
            return ac.act(s.float().to(device), o.float().to(device),
                      deterministic)

    def test_agent():
        ep_ret, ep_len = 0, 0
        for j in tqdm(range(num_test_episodes)):
            #o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            s, o = test_env.reset()
            d, ep_ret, ep_len = False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                a_coded  = get_action(s, o, False)
                _, a = decodeActions(*[a_coded[:, i] for i in range(n_a)])
                s, o, r, d = test_env.step(a)
                ep_ret += r
                ep_len += 1
        return ep_ret, ep_len

    ac.train()


    # following Dians code really closely for the pretrainin
    counter = 0
    update_counter = 0
    if pretrain_episodes > 0:
        planner_envs = env

        planner_num_process = num_processes
        j = 0
        states, obs = planner_envs.reset()
        s = 0

        #if not no_bar:
        planner_bar = tqdm(total=pretrain_episodes)
        while j < pretrain_episodes:
            plan_actions = planner_envs.getNextAction()
            # it doesn't matter what agent we use
            with torch.no_grad():
                planner_actions_star_idx, planner_actions_star = getActionFromPlan(plan_actions)
                states_, obs_, rewards, dones = planner_envs.step(planner_actions_star, auto_reset=True)
            #steps_lefts = planner_envs.getStepLeft()

            # then iterate through eah environment
            for i in range(planner_num_process):
                transition = ExpertTransition(states[i].numpy(), obs[i].numpy(), planner_actions_star_idx[i].numpy(),
                                              rewards[i].numpy(), states_[i].numpy(), obs_[i].numpy(), 
                                              dones[i],np.array(100), np.array(1))
                transition = normalizeTransition(transition)
                replay_buffer.add(transition)
                counter += 1
                #replay_buffer.store(obs[i], states[i], planner_actions_star_idx[i], rewards[i], obs_[i], states_[i], dones[i])
                #local_transitions.append(transition)
            states = copy.copy(states_)
            obs = copy.copy(obs_)
            #if dones.sum() and rewards.sum():
            #    for t in local_transitions:
            #        # store in the replay buffer
                    # TODO: ADD TRANSITION TO REPLAY BUFFER
            #        replay_buffer.store(t[1], t[0], t[2], t[3], t[5], t[4], t[6])

            #    local_transitions = []
            j += dones.sum().item()
            s += rewards.sum().item()
                #if not no_bar:
            planner_bar.set_description('{:.3f}/{}, AVG: {:.3f}'.format(s, j, float(s) / j if j != 0 else 0))
            planner_bar.update(dones.sum().item())
            # adding data augmentation and weighted buffer?
            #if expert_aug_n > 0:
            #    augmentBuffer(replay_buffer, buffer_aug_type, expert_aug_n)
    pretrain_step = counter // batch_size
    pretrain_step = 0
    if pretrain_step > 0:
        print(counter)
        for i in tqdm(range(pretrain_step)):
            batch = replay_buffer.sample(batch_size)
            #loss, error = agent.update(batch)
            update(data=batch, num_update=update_counter)
            update_counter += 1

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()

    s, o  = env.reset()
    ep_ret, ep_len = torch.zeros(num_processes), torch.zeros(num_processes) 
    last_ret, last_len = 0, 0

    # Main loop: collect experience in env and update/log each epoch
    replay_len=0
    for t in tqdm(range(total_steps)):
        
        with torch.no_grad():
            a_coded = get_action(s, o)
            u_a, a = decodeActions(*[a_coded[:, i] for i in range(n_a)])
        #u_a, a = agent.getEGreedyActions(s, o, 0.0)

        env.stepAsync(a, auto_reset=False)

        # Update handling
        if len(replay_buffer) >= 100:
            batch = replay_buffer.sample(batch_size)
            #loss, error = agent.update(batch)
            update(data=batch, num_update=update_counter)
            update_counter += 1

        # Step the env
        s2, o2, r, d  = envs.stepWait()
        ep_ret += r 
        ep_len += torch.ones(num_processes) 
        
        done_idxes = torch.nonzero(d).squeeze(1)
        if done_idxes.shape[0] != 0:
            reset_states_, reset_obs_ = envs.reset_envs(done_idxes)
            for j, idx in enumerate(done_idxes):
                s2[idx] = reset_states_[j]
                o2[idx] = reset_obs_[j]

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        #print(u_a.cpu().numpy())
        # that isn't based on the agent's state)
        for i in range(num_processes):
            transition = ExpertTransition(s[i].numpy(), o[i].numpy(), u_a[i].numpy(), r[i].numpy(), s2[i].numpy(), o2[i].numpy(), d[i].numpy(), np.array(100), 0)
            #transition = ExpertTransition(s[i], o[i], u_a[i], r[i], s2[i], o2[i], d[i], np.array(100), 0)
            transition = normalizeTransition(transition)
            replay_buffer.add(transition)
            replay_len+=1

        o = copy.copy(o2)
        s = copy.copy(s2)

        # End of trajectory handling
        #if d or (ep_len == max_ep_len):
        #    s, o = env.()
        #    last_ret = ep_ret
        #    last_len = ep_len
         #   ep_ret, ep_len = 0,0

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            #if (epoch % save_freq == 0) or (epoch == epochs):

            # Test the performance of the deterministic version of the agent.

            #last_ret, last_len = test_agent()
            #print('test return', last_ret)
            #print('test length', last_len)

            # Log info about epoch
            # here, we add the episode reward and the global timestep
            #writer.add_scalar("charts/discounted_episodic_return", last_ret, t)
            #writer.add_scalar("charts/episodic_length", last_len, t)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('-id', '--gym_id', type=str, help='Id of the environment that we will use', default='close_loop_block_reaching')
    parser.add_argument('--hid', type=int, default=1024)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('-tr', '--track', type=str2bool, help='Track the performance of the environment', nargs='?', const=False, default=False)
    parser.add_argument('-ne', '--num_envs', type=int, default=1)
    parser.add_argument('-re', '--render', type=str2bool, nargs='?', const=False, default=False)
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
    env_config={'workspace': workspace, 'max_steps': 100, 
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
    envs = EnvWrapper(args.num_envs, simulator, args.gym_id, env_config, planner_config)
    test_envs = EnvWrapper(args.num_envs, simulator, args.gym_id, env_config, planner_config)
    torch.set_num_threads(torch.get_num_threads())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gamma = 0.99
    lr = 1e-3
    dpos = 0.05
    drot = np.pi/8
    obs_type='pixel'
    action_sequence=5
    #agent = SAC(lr=(lr, lr), gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
    #       n_a=action_sequence, tau=1e-2, alpha=1e-2, policy_type='gaussian',
    #      target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type)

    obs_channel=2
    crop_size = 128
    equi_n = 8
    n_hidden = 128 
    initialize=True
    agent = core.MLPActorCritic(hidden_sizes=(n_hidden, n_hidden))

    #actor = EquivariantSACActor((obs_channel, crop_size, crop_size), action_sequence, n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
    #critic = EquivariantSACCritic((obs_channel, crop_size, crop_size), action_sequence, n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)

    #agent.initNetwork(actor, critic)
    sac(envs, test_envs, actor_critic=agent,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, num_processes=args.num_envs, epochs=args.epochs,
        track=args.track, gym_id=args.gym_id, device=device)