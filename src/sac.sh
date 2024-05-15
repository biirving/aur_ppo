#!/bin/bash
encoder_type='equiv'
bc_episodes=5000
batch=64
env='close_loop_block_in_bowl'
track=True

#for t in vit base equiv

job=`sbatch -p gpu \
--gres=gpu:p100:1 \
--time=08:00:00 \
--mem=32GB --output=/work/nlp/b.irving/rl_outputs/sac_$batch'_'$env'_'%j.out \
run_sac.py --encoder_type=$encoder_type --bc_episodes=$bc_episodes --track=$track \
--gym_id=$env | awk '{print $NF}'`
echo $job