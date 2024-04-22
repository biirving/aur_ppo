from src.policies.sacBullet import sacBullet
import torch.nn.functional as F


# add this in after:
"""
class sacBC(sacBullet):
    def __init__(self):
        super().__init__()
    
    def compute_loss_pi(self):
        loss = super().compute_loss_pi()
        a_coded = self.loss_calc_dict['pi']  
        mean = self.loss_calc_dict['mean'] 
        log_pi = self.loss_calc_dict['log_pi'] 
        is_experts = self.loss_calc_dict['is_experts']
        if is_experts.sum():
            policy_loss = F.mse_loss()
        return loss_pi
    
"""