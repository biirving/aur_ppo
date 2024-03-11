#!/bin/bash

for value_coeff in 0.0 0.5
do 
    for entropy_coeff in 0.0 0.1 
    do 
        for learning_rate in 3e-3 1e-4
        do 
            for clip_coeff in 0.0 0.2 0.8
            do
                for num_minibatches in 4 16
                do
                    job=$(sbatch -p gpu \
                    --time=08:00:00 \
                    --mem=32GB \
                    --gres=gpu:p100:1 \
                    --output=/work/nlp/b.irving/ppo_outputs/${value_coeff}_${entropy_coeff}_${learning_rate}_${clip_coeff}_${num_minibatches}_%j.out \
                    robot_run.py \
                     --track=True \
                     --value_coeff=$value_coeff \
                     --clip_coeff=$clip_coeff \
                     --entropy_coeff=$entropy_coeff \
                     --learning_rate=$learning_rate \
                     --num_minibatches=$num_minibatches | awk '{print $NF}')
                    echo $job
                done
            done
        done
    done
done
