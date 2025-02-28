# Belief Update BEnchmark
This folder contains all the necessary code required to implement the benchmark algorithm from the article titled 'Infinite Horizon Maximum Causal Entropy Inverse Reinforcement Learning'.

### Contents 
The following files are contained in this folder,

- Code Files:
  - *IHMCE_MaxEnt_Robotarium.ipynb*: This code implements the Algorithms from the article titled 'Infinite Horizon Maximum Causal Entropy Inverse Reinforcement Learning' on the discrete Robotarium.
  - *optimizer.py*: This file implements the gradient ascent based optimization.

### IHMCE_MaxEnt_Comparison
- Infinite Horizon Maximum Causal Entropy IRL:
The first part of the code implements the maximum discounted causal Entropy (MDCE) Algorithm of the article 'Infinite Horizon Maximum Causal Entropy Inverse Reinforcement Learning' on the discrete case pendulum. We define the class for soft-value iteration with 0.99 as the discount factor. We define the cost feature set $\textbf{h}(\textbf{x}_{k})$ as defined above. We define the function to calculate the expected feature value of the policy with actual and the policy with estimated cost. To compute the expected feature value for the estimated cost we utilise Monte-Carlo sampling with a sample size of 100. We use the expected feature value functions to optimise the corresponding feature weights using gradient descent.

- Maximum Causal Entropy IRL:
The Second part of the code implements the Algorithm of the article 'Maximum Entropy Inverse Reinforcement Learning' on the discrete case pendulum. We replace the backward pass of the MaxEnt algorithm with value iteration and define the class for value iteration with 0.9 as the discount factor. We define the cost feature set $\textbf{h}(\textbf{x}_{k})$ as defined above. We define the function to calculate the expected feature value of the policy with actual and the policy with estimated cost. To compute the forward pass and obtain expected feature value for estimated. We use the expected feature value of the expert policy and learner policy to optimise the corresponding feature weights using gradient descent.


Note: We implemented the benchmarking algorithms (MaxEnt and IHMCE) as efficiently as possible. 

$\textbf{Experiment settings:}$
- Initial learning rate: 1.0 (we use an exponential decay function to update the learning rate after each iteration)
- Discount factor for soft-value iteration: 0.99
- Initial weights $\textbf{w}$ are sampled from a uniform distribution with support on $[-100,100]$ (we conducted 10 different experiment with 10 distinct initial conditions) 
- Gradient descent stopping threshold:= 0.001
