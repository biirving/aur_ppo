from abc import ABC, abstractmethod
import torch

class policy(ABC):
    """
    Base class for all policies. This class allows the user a basic structure, should they want it.
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_calc_dict = {}

    @abstractmethod
    def load_info(self):
        """
        Load necessary information or data required by the policy.
        """
        pass
    
    @abstractmethod
    def _loadBatchToDevice(self):
        """
        Load a batch of data to the device. This method should handle any necessary
        transformations to the data before computation.
        """
        pass
    
    @abstractmethod
    def initNet(self):
        """
        Initialize the neural network or any computational graph used by the policy.
        """
        pass

    @abstractmethod
    def update(self):
        """
        Update the policy based on feedback or new data.
        """
        pass

    @abstractmethod
    def act(self):
        """
        Act! Act!
        """
        pass

    @abstractmethod
    def save_agent(self):
        """
        Save our agent
        """
        pass