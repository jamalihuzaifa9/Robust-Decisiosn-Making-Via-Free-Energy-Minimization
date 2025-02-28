# Experiments

This folder contains the code required to perform the robot routing experiments as given in the manuscript ([see this link](https://arxiv.org/abs/2306.13928) for the preprint).
### Prerequisites
To run the code, the first step is to download and install the [robotarium python simulator package](https://github.com/robotarium/robotarium_python_simulator).
### Contents 
The **Experiments** folder contains the following files:

- Code Files:
  - *DR_robot_routing_simulate.py*: The code file implements the DR-FREE Algorithm and performs the robot routing task. The code can be submitted to the robotarium platform. 
  - *DR_robot_routing_IOC.ipynb*: The notebook implements the robot routing experiment by solving the forward and inverse using the algorithms given in the manuscript.
  - *GP_Model_Training.ipynb*: The notebook contains code to train GP models.
  -  *eta_policy.ipynb*: The notebook evaluates the policy computation of DR-FREE algorithm under varying ambiguity radius.
  - *Robo_Dataset_Generate.py*: The code file generates data for training GP models.
- Binaries:
  - *GP_nominal_1.dump*: Stores GP model for training stage 1.
  - *GP_nominal_2.dump*: Stores GP model for training stage 2.
  - *GP_nominal.dump*: Stores GP model for training stage 3.
  - *Weights_DR.npy*: Stores the weights obtained for the reconstructed cost that can replicate the results in the manuscript.

### robot_routing_simulate.py

The file **DR_robot_routing_simulate.py** implements the simulation environment for the DR-FREE framework on a robot routing task. It loads a pre-trained Gaussian Process model to predict nominal state transitions and defines dynamic models, cost functions, and robust control steps that integrate obstacle avoidance, goal attainment, and environmental uncertainties. Leveraging the Robotarium simulation tools the script simulates robot navigation within a bounded workspace populated with obstacles and boundaries. Throughout the simulation, it records state trajectories and control inputs, which are then saved for further analysis.

- The file also implements an ambiguity-unaware agent, the control algorithm can be switched by commenting out the DR-FREE algorithm at lines 318 and 319, and uncommenting lines 322.

### DR_robot_routing_IOC.ipynb

- Forward Problem:
Given the setup of the experiment in the manuscript, the first part of the code solves the forward problem/task of robot routing while avoiding obstacles using the control policy computed by the DR-FREE Algorithm of the manuscript. Multiple state and control input trajectories of a robot performing the task are obtained and saved in the *State_Data.npy* and *Input_Data.npy*. 

- Inverse Problem:
The second part of the code uses these data files to estimate the cost of the agent using Algorithm 2 of the manuscript. 
We define a function that forms the feature vector. We used a 16-dimensional features vector, with the first feature being, $$(x_{k}-x_{d})^{2}$$  the distance from the desired location of the robot,
and with the other features being Gaussians of the form $g_{i}(\textbf{x}_{k})$, centered around 15 uniformly distributed points in the Robotarium work area. 
Next, we obtain the *Weights_DR.npy* by solving the inverse problem. The figure below shows the placement of the feature points on the Robotarium work area with corresponding weight values.
![feature_point_grid](https://github.com/user-attachments/assets/6343edfe-0184-40e6-adbe-07ec5cc66e04)
We use the weights to formulate the estimated cost and test the effectiveness of the estimated cost by performing the robot routing cost while avoiding obstacles. The plots in Figure 3 of the manuscripts can be obtained from the last section of the code.

Note: To replicate Figure 3 of the manuscript use the *Weights_Obtained.npy*, *State_Data*, and *Input_Data* files.
