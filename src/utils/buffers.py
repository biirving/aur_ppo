import numpy as np
import torch
import collections
from scipy.ndimage import affine_transform
import numpy.random as npr
from src.utils.misc import ExpertTransition, ExpertTransitionOffline, ExpertTransitionPPO

def get_image_transform(theta, trans, pivot=(0, 0)):
    """Compute composite 2D rigid transformation matrix."""
    # Get 2D rigid transformation matrix that rotates an image by theta (in
    # radians) around pivot (in pixels) and translates by trans vector (in
    # pixels)
    pivot_t_image = np.array([[1., 0., -pivot[0]], [0., 1., -pivot[1]],
                              [0., 0., 1.]])
    image_t_pivot = np.array([[1., 0., pivot[0]], [0., 1., pivot[1]],
                              [0., 0., 1.]])
    transform = np.array([[np.cos(theta), -np.sin(theta), trans[0]],
                          [np.sin(theta), np.cos(theta), trans[1]], [0., 0., 1.]])
    return np.dot(image_t_pivot, np.dot(transform, pivot_t_image))

def get_random_image_transform_params(image_size):
    theta = np.random.random() * 2*np.pi
    trans = np.random.randint(0, image_size[0]//10, 2) - image_size[0]//20
    pivot = (image_size[1] / 2, image_size[0] / 2)
    return theta, trans, pivot

def perturb(current_image, next_image, dxy, set_theta_zero=False, set_trans_zero=False):
    image_size = current_image.shape[-2:]

    # Compute random rigid transform.
    theta, trans, pivot = get_random_image_transform_params(image_size)
    if set_theta_zero:
        theta = 0.
    if set_trans_zero:
        trans = [0., 0.]
    transform = get_image_transform(theta, trans, pivot)
    transform_params = theta, trans, pivot

    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_dxy = rot.dot(dxy)
    rotated_dxy = np.clip(rotated_dxy, -1, 1)

    # Apply rigid transform to image and pixel labels.
    current_image = affine_transform(current_image, np.linalg.inv(transform), mode='nearest', order=1)
    if next_image is not None:
        next_image = affine_transform(next_image, np.linalg.inv(transform), mode='nearest', order=1)

    return current_image, next_image, rotated_dxy, transform_params
    

def augmentTransitionSO2(d):
    obs, next_obs, dxy, transform_params = perturb(d.obs[0].copy(),
                                                   d.next_obs[0].copy(),
                                                   d.action[1:3].copy(),
                                                   set_trans_zero=True)
    obs = obs.reshape(1, *obs.shape)
    next_obs = next_obs.reshape(1, *next_obs.shape)
    action = d.action.copy()
    action[1] = dxy[0]
    action[2] = dxy[1]
    return ExpertTransition(d.state, obs, action, d.reward, d.next_state,
                            next_obs, d.done, d.step_left, d.expert)

class QLearningBuffer:
    def __init__(self, size, transition_type='base'):
        self._storage = []
        self._max_size = size
        self._next_idx = 0
        self.transition_type='base'

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
    
    def reset(self):
        self._storage=[]
        self._next_idx=0

    def save_buffer(self, save_path, run_id=0):
        # create the numpy arrays from storage
        states = np.stack([d.state for d in self._storage])
        obs = np.stack([d.obs for d in self._storage])
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(1)
        actions = np.stack([d.action for d in self._storage])
        rewards = np.stack([d.reward.squeeze() for d in self._storage])
        dones = np.stack([d.done for d in self._storage])
        steps_left = np.stack([d.step_left for d in self._storage])
        np.save(save_path + '/' + str(run_id) + '_states.npy', states)
        np.save(save_path + '/' + str(run_id) + '_obs.npy', obs)
        np.save(save_path + '/' + str(run_id) + '_actions.npy', actions)
        np.save(save_path + '/' + str(run_id) + '_rewards.npy', rewards)
        np.save(save_path + '/' + str(run_id) + '_dones.npy', dones)
        np.save(save_path + '/' + str(run_id) + '_steps_left.npy', steps_left)
        # transition type specific
        if self.transition_type == 'base':
            next_states = np.stack([d.next_state for d in self._storage])
            next_obs = np.stack([d.next_obs for d in self._storage])
            if len(next_obs.shape) == 3:
                next_obs = next_obs.unsqueeze(1)
            is_experts = np.stack([d.expert for d in self._storage])
            np.save(save_path + '/' + str(run_id) + '_next_states.npy', next_states)
            np.save(save_path + '/' + str(run_id) + '_next_obs.npy', next_obs)
            np.save(save_path + '/' + str(run_id) + '_is_experts.npy', is_experts)
        if self.transition_type == 'offline':
            next_states = np.stack([d.next_state for d in self._storage])
            next_obs = np.stack([d.next_obs for d in self._storage])
            is_experts = np.stack([d.expert for d in self._storage])
            expert_actions = np.stack([d.expert_action for d in self._storage])
            np.save(save_path + '/' + str(run_id) + '_next_states.npy',next_states)
            np.save(save_path + '/' + str(run_id) + '_next_obs.npy', next_obs)
            np.save(save_path + '/' + str(run_id) + '_is_experts.npy', is_experts)
            np.save(save_path + '/' + str(run_id) + '_expert_actions.npy', expert_actions)
        if self.transition_type == 'ppo':
            expert_actions = np.stack([d.expert_action for d in self._storage])
            log_probs = np.stack([d.log_probs for d in self._storage])
            values = np.stack([d.value for d in self._storage])
            np.save(save_path + '/' + str(run_id) + '_expert_actions.npy', expert_actions)
            np.save(save_path + '/' + str(run_id) + '_log_probs.npy', log_probs)
            np.save(save_path + '/' + str(run_id) + '_values.npy', values)
        
    def load_buffer(self, load_path, run_id=0):
        """
        Loading offline transitions from memory, primarily for offline reinforcement learning
        """
        states = np.load(load_path + '/' + str(run_id) + '_states.npy', states)
        obs = np.load(load_path + '/' + str(run_id) + '_obs.npy', obs)
        actions = np.load(load_path + '/' + str(run_id) + '_actions.npy', actions)
        rewards = np.load(load_path + '/' + str(run_id) + '_rewards.npy', rewards)
        dones = np.load(load_path + '/' + str(run_id) + '_dones.npy', dones)
        steps_left = np.load(load_path + '/' + str(run_id) + '_steps_left.npy', steps_left)
        # transition type specific
        if self.transition_type == 'base':
            states_ = np.load(load_path + '/' + str(run_id) + '_next_states.npy', next_states)
            obs_ = np.load(load_path + '/' + str(run_id) + '_next_obs.npy', next_obs)
            is_experts = np.load(load_path + '/' + str(run_id) + '_is_experts.npy', is_experts)
            # fill our storage buffer
            for i in range(states_.shape[0]):
                transition = ExpertTransition(states[i], obs[i], actions[i],
                                              rewards[i], states_[i], obs_[i], 
                                              dones[i], steps_left[i], is_experts[i])
                self.add(transition)
        if self.transition_type == 'offline':
            next_states = np.load(load_path + '/' + str(run_id) + '_next_states.npy', next_states)
            next_obs = np.load(load_path + '/' + str(run_id) + '_next_obs.npy', next_obs)
            is_experts = np.load(load_path + '/' + str(run_id) + '_is_experts.npy', is_experts)
            expert_actions = np.load(load_path + '/' + str(run_id) + '_expert_actions.npy', expert_actions)
            for i in range(states_.shape[0]):
                transition = ExpertTransitionOffline(states[i], obs[i], actions[i],
                                              rewards[i], states_[i], obs_[i], 
                                              dones[i], steps_left[i], is_experts[i], expert_actions[i])
                self.add(transition)
        if self.transition_type == 'ppo':
            expert_actions = np.load(load_path + '/' + str(run_id) + '_expert_actions.npy', expert_actions)
            log_probs = np.load(load_path + '/' + str(run_id) + '_log_probs.npy', log_probs)
            values = np.load(load_path + '/' + str(run_id) + '_values.npy', values)
            for i in range(states.shape[0]):
                transition = ExpertTransitionPPO(states[i], obs[i], actions[i], rewards[i],
                                              dones[i], steps_left[i], expert_actions[i], log_probs[i], values[i])
                self.add(transition)

class QLearningBufferAug(QLearningBuffer):
    def __init__(self, size, aug_n=4):
        super().__init__(size)
        self.aug_n = aug_n

    def add(self, transition: ExpertTransition):
        super().add(transition)
        for _ in range(self.aug_n):
            super().add(augmentTransitionSO2(transition))
