# Equivariant PPO with Immitation Learning

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

## Cite

BulletArm
```
@misc{https://doi.org/10.48550/arxiv.2205.14292,
      doi = {10.48550/ARXIV.2205.14292},
      url = {https://arxiv.org/abs/2205.14292},
      author = {Wang, Dian and Kohler, Colin and Zhu, Xupeng and Jia, Mingxi and Platt, Robert},
      keywords = {Robotics (cs.RO), FOS: Computer and information sciences, FOS: Computer and information sciences},
      title = {BulletArm: An Open-Source Robotic Manipulation Benchmark and Learning Framework},
      publisher = {arXiv},
      year = {2022},
      copyright = {arXiv.org perpetual, non-exclusive license}
}
```

Proximal Policy Optimization

```
@article{DBLP:journals/corr/SchulmanWDRK17,
  author       = {John Schulman and
                  Filip Wolski and
                  Prafulla Dhariwal and
                  Alec Radford and
                  Oleg Klimov},
  title        = {Proximal Policy Optimization Algorithms},
  journal      = {CoRR},
  volume       = {abs/1707.06347},
  year         = {2017},
  url          = {http://arxiv.org/abs/1707.06347},
  eprinttype    = {arXiv},
  eprint       = {1707.06347},
  timestamp    = {Mon, 13 Aug 2018 16:47:34 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/SchulmanWDRK17.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

Equivariant Reinforcement Learning

```
@inproceedings{
wang2022so2equivariant,
title={{$\mathrm{SO}(2)$}-Equivariant Reinforcement Learning},
author={Dian Wang and Robin Walters and Robert Platt},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=7F9cOhdvfk_}
}
```
```
@inproceedings{
wang2022onrobot,
title={On-Robot Learning With Equivariant Models},
author={Dian Wang and Mingxi Jia and Xupeng Zhu and Robin Walters and Robert Platt},
booktitle={6th Annual Conference on Robot Learning},
year={2022},
url={https://openreview.net/forum?id=K8W6ObPZQyh}
}
```

