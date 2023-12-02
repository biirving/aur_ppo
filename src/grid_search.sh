#!/bin/bash


# search for the best PPO parameters

# run PPO without updating weights and biases charts
time_steps=1700000
for batch in 1 4 8 16 32 64
do
for layer in 2 8 16 32
do 
for env in 4 8 16 32
do 
for dim in 64 128 256
do 
for dropout in 0.0 0.2 0.4 0.8
do
#job=$(sbatch -p gpu --time=08:00:00 --mem=32GB --gres=gpu:p100:1 --output=/work/nlp/b.irving/ppo_outputs/$batch'_'$env'_'$dim'_'$dropout'_'%j.out run.py -nm $batch -nl $layer -ne $env -d $dim -do $dropout -t=$time_steps)
job=`sbatch -p short \
--time=08:00:00 \
--mem=32GB \
 --output=/work/nlp/b.irving/ppo_outputs/$batch'_'$env'_'$dim'_'$dropout'_'%j.out \
 run.py \
 -nm $batch \
 -nl $layer \
 -ne $env \
 -d $dim \
 -do $dropout \
 --t=$time_steps | awk '{print $NF}'`
echo $job
done
done
done
done
done