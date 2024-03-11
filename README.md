# aur_ppo

The following repository contains code for Equivariant Proximal Policy Optimization (PPO). The PPO implementation is meant to run on the BulletARM environment and Openai gym.

## Use

To run the default PPO algorithm on the cartpole gym environment, run the following command:

```
python run.py --gym_id='CartPole-v1' --continuous=False
```

Please be sure to set the continuous flag correctly depending on which environment you choose to run.

To run the PPO algorithm on the close_loop_block_picking BulletARM environment with equivariance run the following command:

```
python robot_run.py --gym_id='close_loop_block_picking' --continuous=True --equivariant=True
```

