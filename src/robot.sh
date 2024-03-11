#!/bin/bash

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