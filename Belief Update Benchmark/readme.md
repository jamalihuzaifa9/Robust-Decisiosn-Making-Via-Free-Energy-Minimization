# Belief Update BEnchmark
This folder contains all the necessary code required to implement the benchmark algorithm from the article titled 'Infinite Horizon Maximum Causal Entropy Inverse Reinforcement Learning'.

### Contents 
The following files are contained in this folder,

- Code Files:
  - *IHMCE_MaxEnt_Robotarium.ipynb*: This code implements the Algorithms from the article titled 'Infinite Horizon Maximum Causal Entropy Inverse Reinforcement Learning' on the discrete Robotarium.
  - *optimizer.py*: This file implements the gradient ascent-based optimization.

### IHMCE_MaxEnt_Comparison
- Infinite Horizon Maximum Causal Entropy IRL:
The code implements the maximum discounted causal Entropy (MDCE) Algorithm of the article 'Infinite Horizon Maximum Causal Entropy Inverse Reinforcement Learning' on the discrete case Robotarium. We define the class for soft-value iteration with 0.9 as the discount factor. We define the cost feature set as defined in the manuscript. We define the function to calculate the expected feature value of the policy with actual and the policy with estimated cost. To compute the expected feature value for the estimated cost we utilize Monte-Carlo sampling with a sample size of 100. We use the expected feature value functions to optimize the corresponding feature weights using gradient ascent.

Note: We implemented the benchmarking algorithm (IHMCE) as efficiently as possible. 

$\textbf{Experiment settings:}$
- Initial learning rate: 1.0 (we use an exponential decay function to update the learning rate after each iteration)
- Discount factor for soft-value iteration: 0.9
- Initial weights $\textbf{w}$ are sampled from a uniform distribution with support on $[-100,100]$ 
- Gradient ascent stopping threshold:= 0.001
