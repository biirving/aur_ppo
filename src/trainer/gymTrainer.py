from src.trainer.policyTrainer import policyTrainer
import numpy as np
from src.utils.buffers import QLearningBuffer, QLearningBufferAug
from src.utils.misc import ExpertTransitionPPO, normalizeTransition, store_returns
from src.utils.env_wrapper import EnvWrapper
from tqdm import tqdm
import copy
import gym

class gymTrainer(policyTrainer):
    def __init__(self,  total_time_steps, num_env_steps, num_processes, save_path=None, aug=False, do_pretraining=True, track=False, run_id=0, transition_type='base', capture_video=False):
        super().__init__(track, run_id)
        self.aug=aug
        self.num_env_steps = num_env_steps
        self.total_time_steps = total_time_steps
        #if aug:
        #    self.replay_buffer = QLearningBufferAug(total_time_steps)
        #else:
        self.replay_buffer = QLearningBuffer(total_time_steps, transition_type=transition_type)
        self.num_processes = num_processes
        self.do_pretraining = do_pretraining
        self.returns = store_returns(num_processes)
        self.eval_returns = store_returns(num_processes)
        self.num_eval_processes=1
        self.save_path=save_path
        self.capture_video=capture_video

    def initialize_env(self, simulator, env_config, planner_config, gym_id):
        self.envs = gym.vector.SyncVectorEnv(
	    	[self.make_env(gym_id, i, self.capture_video) for i in range(self.num_processes)]
	    )
        self.eval_envs = self.envs

    def make_env(self, gym_id, idx, capture_video):
		def thunk():
			env = gym.make(gym_id)
			env = gym.wrappers.RecordEpisodeStatistics(env)
			if capture_video:
				if idx == 0:
					env = gym.wrappers.RecordVideo(env, f"videos/{self.run_name}")
			if(self.continuous):
				env = gym.wrappers.ClipAction(env)
				env = gym.wrappers.NormalizeObservation(env)
				env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
				env = gym.wrappers.NormalizeReward(env)
				env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
			return env
		return thunk