# MaxDiff Benchmark

This folder contains the code files used for replicating the **MaxDiff Reinforcement Learning Benchmark** results.

> 🔬 This code builds upon the original implementation from the paper:  
> **[Maximum Diffusion Reinforcement Learning](https://arxiv.org/html/2309.15293v4)**

---

## Main Algorithm

- **MaxDiff3.py**  
  Implements the MaxDiff RL algorithm based on the path entropy formulation and trajectory-level KL divergence minimization.

---

## Robotarium Integration

- **Env_Robotarium.py**  
  Defines the custom Robotarium simulation environment.
- **Maxdiff_Robotarium.py**  
  Runs MaxDiff RL in the Robotarium environment.
---

## MaxDiff Variants and Experiments

- **Maxdiff.py**  
  An earlier baseline version of MaxDiff.
- **Maxdiff2.py**  
  Intermediate implementation variant (possibly with alternate objective or structure).
- **maxdiff_simple_exp.py**  
  A simple experimental script for quick verification of the MaxDiff implementation.
- **maxdiff_eval.py**  
  Evaluation script to compute performance metrics from rollout logs.

---

## Supporting Modules

- **entropy.py**  
  Computes local entropy terms (e.g., `log det C[x]`) used in MaxDiff exploration.
- **maxdiff_utils.py**  
  Utility functions for trajectory generation, reward shaping, and entropy computation.
- **utils.py**  
  General-purpose helper functions shared across scripts.

---

## Learning Components

- **nn_model.py**  
  Defines the neural network models used for dynamics prediction or reward estimation.
- **optimizer.py**  
  Configures optimizer settings and learning rate schedules.
- **replaybuffer.py**  
  Stores agent transitions for experience replay.
---
For theoretical background and derivations, please refer to the official MaxDiff paper linked above.
