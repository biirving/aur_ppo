#!/bin/bash
encoder_type='equiv'
bc_episodes=100
batch=64
gym_id='close_loop_block_in_bowl'
track=True
algo='sacoffline'

random_number=$((RANDOM+100000))
run_id=$((random_number % 1000000))

job=`sbatch -p gpu \
--gres=gpu:p100:1 \
--time=08:00:00 \
--mem=32GB --output=/work/nlp/b.irving/rl_outputs/sac_$batch'_'$env'_'%j.out \
run.py --encoder_type=$encoder_type --bc_episodes=$bc_episodes --track=$track --algo=$algo \
--gym_id=$gym_id --run_id=$run_id | awk '{print $NF}'`
echo $job