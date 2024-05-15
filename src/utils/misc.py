import numpy as np
import collections

# TODO: Should be defined base on learning algorithm
ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done step_left expert')
ExpertTransitionOffline = collections.namedtuple('ExpertTransitionOffline', 'state obs action reward next_state next_obs done step_left expert expert_action')
ExpertTransitionPPO = collections.namedtuple('ExpertTransitionPPO', 'state obs action reward done step_left expert_action log_probs value')

ExpertTransitionGym = collections.namedtuple('ExpertTransitionGym', 'obs action reward done log_probs value')

def normalize_observation(obs):
    """Normalize observation data."""
    obs = np.clip(obs, 0, 0.32)
    obs = obs / 0.4 * 255
    return obs.astype(np.uint8)

def normalizeTransition(d):
    """General function to normalize transitions that works with different types of ExpertTransition."""
    new_obs = normalize_observation(d.obs)

    if hasattr(d, 'next_obs') and hasattr(d, 'expert_action'):
        # Normalizing next_obs only if it exists
        new_next_obs = normalize_observation(d.next_obs)
        return ExpertTransition(d.state, new_obs, d.action, d.reward, d.next_state, new_next_obs, d.done, d.step_left, d.expert)
    
    # don't necessarily need to normalize our transitions
    elif hasattr(d, 'expert_action'):
        # For ExpertTransitionPPO which does not have next_obs
        return ExpertTransitionGym(d.obs, d, action, d.reward, d.done, d.log_probs, d.value)

def normalizeTransitionOffline(d: ExpertTransitionOffline):
    obs = np.clip(d.obs, 0, 0.32)
    obs = obs/0.4*255
    obs = obs.astype(np.uint8)
    next_obs = np.clip(d.next_obs, 0, 0.32)
    next_obs = next_obs/0.4*255
    next_obs = next_obs.astype(np.uint8)
    return ExpertTransitionOffline(d.state, obs, d.action, d.reward, d.next_state, next_obs, d.done, d.step_left, d.expert, d.expert_action)

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