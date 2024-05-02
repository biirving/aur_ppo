from abc import ABC, abstractmethod
import torch
import numpy as np

class policyTrainer(ABC):
    def __init__(self, track=False, run_id=0):
        '''
        Template for a general policy trainer.
        '''
        self.device = torch.device('cuda')
        self.track=track
        self.run_id=run_id

    def set_threads_and_seeds(self, seed=0):
        # intra op parallelism on cpu
        torch.set_num_threads(torch.get_num_threads())
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True 
    
    @abstractmethod
    def initialize_env(self):
        '''
        Initialize our environment to run our actor critic algorithm.
        '''
        pass

    @abstractmethod
    def pretrain(self):
        '''
        Pretrain our policy.
        '''
        pass

    @abstractmethod
    def evaluate(self):
        '''
        Evaluate our policy.
        '''
        pass

    @abstractmethod
    def step_env(self):
        '''
        Step our environment forward, fill necessary buffers, and chart our progress
        '''
        pass

    @abstractmethod
    def run(self):
        '''
        Run the training loop for our policy
        '''
        pass