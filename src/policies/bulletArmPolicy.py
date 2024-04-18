import torch
from src.policies.policy import policy
import numpy as np

class bulletArmPolicy(policy):
    def __init__(self, dr=8, dx=0.05, dy=0.05, dz=0.05, n_a=5, obs_type='pixel'):
        """
        The bulletArmPolicy class is an abstract structure meant to encapsulate policies
        designed to run in the BulletArm environment. Many of the methods used below
        draw from the BulletArm repository https://github.com/ColinKohler/BulletArm
        """
        super().__init__()
        self.n_a=n_a
        self.p_range = torch.tensor([0, 1])
        self.dtheta_range = torch.tensor([-np.pi/dr, np.pi/dr])
        self.dx_range = torch.tensor([-dx, dx])
        self.dy_range = torch.tensor([-dy, dy])
        self.dz_range = torch.tensor([-dz, dz])	
        self.obs_type = obs_type

    def _loadBatchToDevice(self, batch, device='cuda'):

        states_tensor = torch.tensor(np.stack([d.state for d in batch])).long().to(device)
        obs_tensor = torch.tensor(np.stack([d.obs for d in batch])).to(device)
        if len(obs_tensor.shape) == 3:
            obs_tensor = obs_tensor.unsqueeze(1)
        action_tensor = torch.tensor(np.stack([d.action for d in batch])).to(device)
        rewards_tensor = torch.tensor(np.stack([d.reward.squeeze() for d in batch])).to(device)
        next_states_tensor = torch.tensor(np.stack([d.next_state for d in batch])).long().to(device)
        next_obs_tensor = torch.tensor(np.stack([d.next_obs for d in batch])).to(device)
        if len(next_obs_tensor.shape) == 3:
            next_obs_tensor = next_obs_tensor.unsqueeze(1)
        dones_tensor = torch.tensor(np.stack([d.done for d in batch])).int()
        non_final_masks = (dones_tensor ^ 1).float().to(device)
        step_lefts_tensor = torch.tensor(np.stack([d.step_left for d in batch])).to(device)
        is_experts_tensor = torch.tensor(np.stack([d.expert for d in batch])).bool().to(device)

        # scale observation from int to float
        obs_tensor = obs_tensor/255*0.4
        next_obs_tensor = next_obs_tensor/255*0.4

        self.loss_calc_dict['batch_size'] = len(batch)
        self.loss_calc_dict['states'] = states_tensor
        self.loss_calc_dict['obs'] = obs_tensor
        self.loss_calc_dict['action_idx'] = action_tensor
        self.loss_calc_dict['rewards'] = rewards_tensor
        self.loss_calc_dict['next_states'] = next_states_tensor
        self.loss_calc_dict['next_obs'] = next_obs_tensor
        self.loss_calc_dict['non_final_masks'] = non_final_masks
        self.loss_calc_dict['step_lefts'] = step_lefts_tensor
        self.loss_calc_dict['is_experts'] = is_experts_tensor
        
        return states_tensor, obs_tensor, action_tensor, rewards_tensor, next_states_tensor, \
               next_obs_tensor, non_final_masks, step_lefts_tensor, is_experts_tensor

    def load_info(self):
        """
        Get the loaded batch data in self.loss_calc_dict
        :return: batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts
        """
        batch_size = self.loss_calc_dict['batch_size']
        states = self.loss_calc_dict['states']
        obs = self.loss_calc_dict['obs']
        action_idx = self.loss_calc_dict['action_idx']
        rewards = self.loss_calc_dict['rewards']
        next_states = self.loss_calc_dict['next_states']
        next_obs = self.loss_calc_dict['next_obs']
        non_final_masks = self.loss_calc_dict['non_final_masks']
        step_lefts = self.loss_calc_dict['step_lefts']
        is_experts = self.loss_calc_dict['is_experts']
        return batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts

    def _loadLossCalcDict(self):
        """
        get the loaded batch data in self.loss_calc_dict
        :return: batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts
        """
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self.load_info() 

        if self.obs_type == 'pixel':
            # stack state as the second channel of the obs
            obs = torch.cat([obs, states.reshape(states.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])], dim=1)
            next_obs = torch.cat([next_obs, next_states.reshape(next_states.size(0), 1, 1, 1).repeat(1, 1, next_obs.shape[2], next_obs.shape[3])], dim=1)

        return batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts

    def decodeActions(self, *args):
        """
        The decodeActions method adjusts an action input for the BulletArm environment. The scaling 
        settings are specific to the BulletArm library. 

        Please see https://github.com/ColinKohler/BulletArm
        """
        unscaled_p = args[0]
        unscaled_dx = args[1]
        unscaled_dy = args[2]
        unscaled_dz = args[3]

        p = 0.5 * (unscaled_p + 1) * (self.p_range[1] - self.p_range[0]) + self.p_range[0]
        dx = 0.5 * (unscaled_dx + 1) * (self.dx_range[1] - self.dx_range[0]) + self.dx_range[0]
        dy = 0.5 * (unscaled_dy + 1) * (self.dy_range[1] - self.dy_range[0]) + self.dy_range[0]
        dz = 0.5 * (unscaled_dz + 1) * (self.dz_range[1] - self.dz_range[0]) + self.dz_range[0]

        if self.n_a == 5:
            unscaled_dtheta = args[4]
            dtheta = 0.5 * (unscaled_dtheta + 1) * (self.dtheta_range[1] - self.dtheta_range[0]) + self.dtheta_range[0]
            actions = torch.stack([p, dx, dy, dz, dtheta], dim=1)
            unscaled_actions = torch.stack([unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz, unscaled_dtheta], dim=1)
        else:
            actions = torch.stack([p, dx, dy, dz], dim=1)
            unscaled_actions = torch.stack([unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz], dim=1)
        return unscaled_actions, actions

    def getActionFromPlan(self, plan):
        """
        Get unscaled and scaled actions from scaled planner action
        :param plan: scaled planner action (in true scale)
        :return: unscaled_actions (in range (-1, 1)), actions (in true scale)
        """
        def getUnscaledAction(action, action_range):
            unscaled_action = 2 * (action - action_range[0]) / (action_range[1] - action_range[0]) - 1
            return unscaled_action

        dx = plan[:, 1].clamp(*self.dx_range)
        p = plan[:, 0].clamp(*self.p_range)
        dy = plan[:, 2].clamp(*self.dy_range)
        dz = plan[:, 3].clamp(*self.dz_range)

        unscaled_p = getUnscaledAction(p, self.p_range)
        unscaled_dx = getUnscaledAction(dx, self.dx_range)
        unscaled_dy = getUnscaledAction(dy, self.dy_range)
        unscaled_dz = getUnscaledAction(dz, self.dz_range)

        if self.n_a == 5:
            dtheta = plan[:, 4].clamp(*self.dtheta_range)
            unscaled_dtheta = getUnscaledAction(dtheta, self.dtheta_range)
            return self.decodeActions(unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz, unscaled_dtheta)
        else:
            return self.decodeActions(unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz)
