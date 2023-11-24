import gymnasium as gym
import panda_gym
env = gym.make('PandaReach-v3') 
observation, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation)
    if terminated or truncated:
        observation, info = env.reset()
env.close()