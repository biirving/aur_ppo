from ppo import ppo
import gymnasium as gym



env = gym.make('CartPole-v1')
state, _ = env.reset()

gamma = 0.9
alpha = 0.1
epsilon = 0.1
state_dim = 4
action_dim = 2
hidden_dim = 64
num_layers = 1
dropout = 0.0
continuous = False 
actor_lr = 5e-5
critic_lr = 5e-5
gym_id = 'CartPole-v1'


params = {
		'gym_id':gym_id,  

		
		self.seed = params['seed']
		self.num_steps = params['num_steps']
		self.max_length = params['max_length']
		self.gae = params['gae']
		self.total_timesteps = params['total_timesteps']
		self.batch_size = params['batch_size']
		self.num_updates = params['num_updates']
		self.anneal_lr = params['anneal_lr']
		self.gae_lambda = params['gae_lambda']
		self.num_update_epochs = params['num_update_epochs']
		self.num_envs = params['num_envs']
		self.num_minibatches = params['num_minibatches']
}

test = ppo(
	)



