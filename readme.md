# Constrained Policy Optimization for rllab

Constrained Policy Optimization (CPO) [1] is an algorithm for learning policies that should satisfy behavioral constraints throughout training. 

This module was designed for [rllab](https://github.com/openai/rllab) [2], and includes the implementations of
- [Constrained Policy Optimization](https://github.com/jachiam/cpo/blob/master/algos/safe/cpo.py)
- [Primal-Dual Optimization](https://github.com/jachiam/cpo/blob/master/algos/safe/pdo.py)
- [Fixed Penalty Optimization](https://github.com/jachiam/cpo/blob/master/algos/safe/fpo.py)

described in our paper [1]. 

To configure, run the following commands in the root folder of `rllab`:

```bash
git submodule add -f git@github.com:jachiam/cpo.git sandbox/cpo
touch sandbox/__init__.py
```

***

1. Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel. "[Constrained Policy Optimization](https://arxiv.org/abs/1705.10528)". _Proceedings of the 34th International Conference on Machine Learning (ICML), 2017._ 
2. Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel. "[Benchmarking Deep Reinforcement Learning for Continuous Control](http://arxiv.org/abs/1604.06778)". _Proceedings of the 33rd International Conference on Machine Learning (ICML), 2016._

