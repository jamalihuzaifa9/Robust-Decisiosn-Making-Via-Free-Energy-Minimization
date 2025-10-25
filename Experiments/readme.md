# Experiments

This folder contains the code required to perform the robot routing experiments as given in the manuscript ([see this link](https://arxiv.org/abs/2306.13928) for the preprint).
### Prerequisites
To run the code, the first step is to download and install the [robotarium python simulator package](https://github.com/robotarium/robotarium_python_simulator).
### Contents 
The **Experiments** folder contains the following files:

- Code Files:
  - *DR_robot_routing_simulate.py*: The code file implements the DR-FREE Algorithm and performs the robot routing task. The code can be submitted to the robotarium platform. 
  - *DR_robot_routing_IOC.ipynb*: The notebook implements the robot routing experiment by solving the forward and inverse using the algorithms given in the manuscript.
  - *FPD_robot_routing_simulate_recursion.py*: The code file implements the FPD Algorithm with backward recursion and performs the robot routing task.
  - *GP_Model_Training.ipynb*: The notebook contains code to train GP models.
  - *eta_policy.ipynb*: The notebook evaluates the policy computation of DR-FREE algorithm under varying ambiguity radius.
  - *Robo_Dataset_Generate.py*: The code file generates data for training GP models.
- Binaries:
  - *GP_nominal_1.dump*: Stores GP model for training stage 1.
  - *GP_nominal_2.dump*: Stores GP model for training stage 2.
  - *GP_nominal.dump*: Stores GP model for training stage 3.
  - *Weights_DR.npy*: Stores the weights obtained for the reconstructed cost that can replicate the results in the manuscript.

### DR_robot_routing_simulate.py

The file **DR_robot_routing_simulate.py** implements the simulation environment for the DR-FREE framework on a robot routing task. It loads a pre-trained Gaussian Process model to predict nominal state transitions and defines dynamic models, cost functions, and robust control steps that integrate obstacle avoidance, goal attainment, and environmental uncertainties. Leveraging the Robotarium simulation tools the script simulates robot navigation within a bounded workspace populated with obstacles and boundaries. Throughout the simulation, it records state trajectories and control inputs, which are then saved for further analysis.

- The file also implements an ambiguity-unaware agent, the control algorithm can be switched by commenting out the DR-FREE algorithm at lines 318 and 319, and uncommenting lines 322.

### DR_robot_routing_IOC.ipynb

- DR-FREE Algorithm:
The first part of the code implements the DR-FREE algorithm given in the manuscript and generates robot trajectory data.

- Belief update:
The second part of the code uses these data files to estimate the cost of the agent using the belief update algorithm of the manuscript. 
We define a function that forms the feature vector.  Next, we obtain the *Weights_DR.npy* by solving the convex belief update problem. The figure below shows the placement of the feature points on the Robotarium work area with corresponding weight values.
![feature_point_grid](https://github.com/user-attachments/assets/f749acb2-1d2f-4234-8e71-0b165b21e832)
We use the weights to formulate the estimated cost and test the effectiveness of the estimated cost by performing the robot routing cost while avoiding obstacles.

### eta_policy.ipynb

The notebook shows how DR-FREE policy changes as a function of the ambiguity radius $(\eta(x_{k-1},u_{k}))$

![Policy_diffusion](https://github.com/user-attachments/assets/1077ab0d-bf87-4afd-805c-1787e19a9595)
Figure. By increasing the radius of ambiguity $(\eta(x_{k-1},u_{k}))$, the DR-FREE policy (left) becomes proportional to the generative model $(q_{k}^{(u)})$ and ambiguity radius $(\eta(x_{k-1},u_{k}))$ (right).

### GP_Model_Training.ipynb

The notebook implements GP training by leveraging the *scikit-learn* library.




# Experiments

This folder contains the code required to perform the robot routing experiments as given in the manuscript ([see this link](https://arxiv.org/abs/2306.13928) for the preprint).

### Prerequisites
To run the code, the first step is to download and install the [robotarium python simulator package](https://github.com/robotarium/robotarium_python_simulator).

### Contents 
The **Experiments** folder contains the following files:

- Code Files:
  - **DR_robot_routing_simulate.py**: The main implementation file for the DR-FREE Algorithm that performs the robot routing task. This file includes:
    - Switchable implementation between DR-FREE and FPD methods using the `method` argument in `Control_step()` (no need to comment/uncomment code)
    - Configurable ambiguity radius (η) parameter within the `Control_step()` function for exploring different robustness levels
    - Implementation of the special case when η=0, which reduces to standard free energy minimization
    - Results for different ambiguity levels presented in the manuscript are generated from this file
    - The code can be submitted to the robotarium platform for real-world experiments
    
  - **DR_robot_routing_simulate_fixed.py**: Updated version with compatibility fixes for Python 3.10+ and matplotlib 3.5+. Use this version if you encounter `Rectangle.__init__()` errors.
  
  - **DR_robot_routing_new_obstacles.py**: Implementation of DR-FREE algorithm with different obstacle configurations. This file demonstrates the robustness of DR-FREE across various challenging obstacle setups including:
    - Dense obstacle fields
    - Narrow passage navigation
    - Complex multi-obstacle scenarios
    - Dynamic obstacle arrangements
    
  - **DR_robot_routing_IOC.ipynb**: The notebook implements the robot routing experiment by solving the forward and inverse problems using the algorithms given in the manuscript. Includes:
    - Forward problem: DR-FREE trajectory generation
    - Inverse problem: Cost reconstruction using belief update algorithm
    
  - **FPD_robot_routing_simulate_recursion.py**: The code file implements the FPD Algorithm with backward recursion and performs the robot routing task. This provides a baseline comparison without distributional robustness.
  
  - **DR_robot_routing_simulate_recursion.py**: DR-FREE implementation with backward recursion for computing cost-to-go values.
  
  - **GP_Model_Training.ipynb**: The notebook contains code to train GP models for different training stages.
  
  - **eta_policy.ipynb**: The notebook evaluates the policy computation of DR-FREE algorithm under varying ambiguity radius (η). Shows how the policy transitions from risk-neutral to risk-averse as η increases.
  
  - **Robo_Dataset_Generate.py**: The code file generates trajectory data for training GP models across different training stages.

- Binaries:
  - **GP_nominal_1.dump**: Stores GP model for training stage 1 (limited data).
  - **GP_nominal_2.dump**: Stores GP model for training stage 2 (moderate data).
  - **GP_nominal.dump**: Stores GP model for training stage 3 (abundant data).
  - **Weights_DR.npy**: Stores the weights obtained for the reconstructed cost that can replicate the results in the manuscript.
  - **State_Dataset_Maxdiff.npy**: Trajectory data used for MaxDiff benchmark comparisons.

### DR_robot_routing_simulate.py

The file **DR_robot_routing_simulate.py** implements the simulation environment for the DR-FREE framework on a robot routing task. It loads a pre-trained Gaussian Process model to predict nominal state transitions and defines dynamic models, cost functions, and robust control steps that integrate obstacle avoidance, goal attainment, and environmental uncertainties. Leveraging the Robotarium simulation tools, the script simulates robot navigation within a bounded workspace populated with obstacles and boundaries. Throughout the simulation, it records state trajectories and control inputs, which are then saved for further analysis.

**Key Features:**
- **Method Selection**: Toggle between DR-FREE (`method='DR'`) and FPD (`method='FPD'`) by changing the argument in `Control_step()` function (line 318-319). No need to comment/uncomment code blocks.
- **Ambiguity Radius Control**: The ambiguity radius (η) can be modulated within the `Control_step()` function to explore different levels of robustness. The results for varying ambiguity levels in the manuscript are generated by adjusting this parameter.
- **Special Case η=0**: Includes implementation of the case when η=0, which corresponds to standard free energy minimization without distributional robustness.
- **Model Selection**: Switch between using the learned GP model (`model_known=False`) or the known nominal model (`model_known=True`).

**Usage Example:**
```python
# Use DR-FREE with GP model
dxi, u_pf = Control_step(x_si, U_space_1, U_space_2, goal_points, obs_points, 
                        method='DR', model_known=False)

# Use FPD with known model
dxi, u_pf = Control_step(x_si, U_space_1, U_space_2, goal_points, obs_points, 
                        method='FPD', model_known=True)
```

### DR_robot_routing_new_obstacles.py

This file extends the DR-FREE implementation to handle various obstacle configurations beyond the standard setup. It demonstrates the algorithm's capability to navigate through:
- Complex obstacle arrangements with multiple static obstacles
- Narrow corridors and tight spaces
- Dense obstacle fields requiring sophisticated path planning
- Scenarios with obstacles placed at critical decision points

The file uses the same DR-FREE core algorithm but applies it to different workspace configurations, validating the robustness and generalization capability of the approach.

### DR_robot_routing_IOC.ipynb

- **DR-FREE Algorithm (Forward Problem)**:
The first part of the code implements the DR-FREE algorithm given in the manuscript and generates robot trajectory data under model uncertainty and environmental ambiguity.

- **Belief Update (Inverse Problem)**:
The second part of the code uses trajectory data to estimate the agent's cost function using the belief update algorithm described in the manuscript. 

**Workflow:**
1. Define a function that forms the feature vector based on workspace discretization
2. Solve the convex belief update problem to obtain optimal weights stored in `Weights_DR.npy`
3. The figure below shows the placement of feature points on the Robotarium workspace with corresponding weight values:

![feature_point_grid](https://github.com/user-attachments/assets/f749acb2-1d2f-4234-8e71-0b165b21e832)

4. Use the learned weights to formulate the estimated cost function
5. Test the effectiveness of the reconstructed cost by performing robot routing tasks while avoiding obstacles

This validates that the belief update algorithm can accurately recover the underlying cost structure from observed trajectories.

### eta_policy.ipynb

The notebook shows how the DR-FREE policy changes as a function of the ambiguity radius η(x_{k-1}, u_k). This analysis demonstrates the trade-off between optimality and robustness:

![Policy_diffusion](https://github.com/user-attachments/assets/1077ab0d-bf87-4afd-805c-1787e19a9595)

**Figure.** By increasing the radius of ambiguity η(x_{k-1}, u_k), the DR-FREE policy (left panel) becomes more conservative and approaches the generative model q_k^{(u)} weighted by the ambiguity radius η(x_{k-1}, u_k) (right panel). This shows how DR-FREE naturally interpolates between risk-neutral (η=0) and risk-averse (large η) behaviors.

### GP_Model_Training.ipynb

The notebook implements Gaussian Process training by leveraging the *scikit-learn* library. It covers:
- Data preprocessing and normalization
- Kernel selection and hyperparameter tuning
- Model training across different data availability scenarios (training stages 1, 2, 3)
- Model validation and performance evaluation
- Saving trained models as `.dump` files for use in simulation

### Running the Experiments

**Basic DR-FREE simulation:**
```bash
python DR_robot_routing_simulate.py
```

**DR-FREE with different obstacle configurations:**
```bash
python DR_robot_routing_new_obstacles.py
```

**FPD baseline with recursion:**
```bash
python FPD_robot_routing_simulate_recursion.py
```

**Generate training data:**
```bash
python Robo_Dataset_Generate.py
```

### Notes

- All simulation files save trajectory data in `.npy` format for post-processing and analysis
- The `show_figure` parameter can be toggled to enable/disable visualization during simulation
- Computation times are logged and saved in `avg_control_times.npy`
- For real-world Robotarium deployment, ensure `sim_in_real_time=True` in the Robotarium initialization
