from src.trainer.policyTrainer import policyTrainer
import numpy as np
from src.utils.buffers import QLearningBuffer, QLearningBufferAug
from src.utils.misc import ExpertTransitionPPO, normalizeTransition, store_returns
from src.utils.env_wrapper import EnvWrapper
from tqdm import tqdm
import copy

class bulletTrainer(policyTrainer):
    def __init__(self,  total_time_steps, num_env_steps, num_processes,  aug=False, do_pretraining=True, track=False):
        super().__init__(track)
        self.aug=aug
        self.num_env_steps = num_env_steps
        self.total_time_steps = total_time_steps
        #if aug:
        #    self.replay_buffer = QLearningBufferAug(total_time_steps)
        #else:
        self.replay_buffer = QLearningBuffer(total_time_steps)
        self.num_processes = num_processes
        self.do_pretraining = do_pretraining
        self.returns = store_returns(num_processes)
        self.eval_returns = store_returns(num_processes)
        self.num_eval_processes=1

    def initialize_env(self, simulator, env_config, planner_config, gym_id):
        self.envs = EnvWrapper(self.num_processes, simulator, gym_id, env_config, planner_config)
        self.eval_envs = EnvWrapper(self.num_eval_processes, simulator, gym_id, env_config, planner_config)
    