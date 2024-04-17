import numpy as np
import collections

# TODO: Should be defined base on learning algorithm
ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done step_left expert')
ExpertTransitionPPO = collections.namedtuple('ExpertTransitionPPO', 'state obs action reward done step_left expert_action log_probs value')

def normalize_observation(obs):
    """Normalize observation data."""
    obs = np.clip(obs, 0, 0.32)
    obs = obs / 0.4 * 255
    return obs.astype(np.uint8)

def normalizeTransition(d):
    """General function to normalize transitions that works with different types of ExpertTransition."""
    new_obs = normalize_observation(d.obs)

    if hasattr(d, 'next_obs'):
        # Normalizing next_obs only if it exists
        new_next_obs = normalize_observation(d.next_obs)
        return ExpertTransition(d.state, new_obs, d.action, d.reward, d.next_state, new_next_obs, d.done, d.step_left, d.expert)
    else:
        # For ExpertTransitionPPO which does not have next_obs
        return ExpertTransitionPPO(d.state, new_obs, d.action, d.reward, d.done, d.step_left, d.expert_action, d.log_probs, d.value)

class store_returns():
    def __init__(self, num_envs, gamma=0.99):
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

    def reset(self):
        self.env_returns = [[] for _ in range(num_envs)]