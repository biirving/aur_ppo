import numpy as np
import collections

# TODO: Should be defined base on learning algorithm
ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done step_left expert')

def normalizeTransition(d: ExpertTransition):
    obs = np.clip(d.obs, 0, 0.32)
    obs = obs/0.4*255
    obs = obs.astype(np.uint8)

    next_obs = np.clip(d.next_obs, 0, 0.32)
    next_obs = next_obs/0.4*255
    next_obs = next_obs.astype(np.uint8)

    return ExpertTransition(d.state, obs, d.action, d.reward, d.next_state, next_obs, d.done, d.step_left, d.expert)

class store_returns():
    def __init__(self, num_envs, gamma):
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