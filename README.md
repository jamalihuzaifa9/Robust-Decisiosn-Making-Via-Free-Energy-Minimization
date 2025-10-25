# Robust Decision-Making Via Free Energy Minimization
### Introduction
This repository collects the supporting code for the manuscript **Robust Decision-Making Via Free Energy Minimization**. The manuscript discusses the challenge of ensuring the robust performance of autonomous agents amidst environmental and training ambiguities. We introduce DR-FREE, an energy-based computational model that embeds robustness directly into decision-making via free energy minimization. By leveraging a distributionally robust extension of the free energy principle, DR-FREE yields policies that are both optimal and resilient against uncertainty, as demonstrated through real rover experiments.

## Installation

### System Requirements
- **Python Version:** 3.10
- **Operating Systems:** 
  - Ubuntu 22.04 LTS (tested)
  - Windows 10/11 (tested)
- **Hardware:** CUDA-capable GPU (optional, for faster training)

### Prerequisites
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git

### Setup Instructions

#### Option 1: Using Conda Environment File (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Robust-Decisiosn-Making-Via-Free-Enerrgy-Minimization.git
cd Robust-Decisiosn-Making-Via-Free-Enerrgy-Minimization
```

2. Create the conda environment from the provided YAML file:
```bash
conda env create -f drfree_environment.yml
```

3. Activate the environment:
```bash
# On Linux/MacOS
conda activate drfree

# On Windows
conda activate drfree
```

#### Option 2: Using pip and requirements.txt

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Robust-Decisiosn-Making-Via-Free-Enerrgy-Minimization.git
cd Robust-Decisiosn-Making-Via-Free-Enerrgy-Minimization
```

2. Create a Python 3.10 virtual environment:
```bash
# On Linux/MacOS
python3.10 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
```

3. Upgrade pip and install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```


#### Install Roboptarium Pyhton Simulator
```bash
# Clone the robotarium simulator repository
git clone https://github.com/robotarium/robotarium_python_simulator.git

# Navigate to the simulator directory
cd robotarium_python_simulator

# Install the simulator
pip install -e .

# Return to the main project directory
cd ..
```

### Contents
The following list of directories can be found in the repository, reproducing the simulation and experimental results in the manuscript.
- Experiments:
  - The folder contains our DRFREE implementation for the Robotarium experiments.
  - The folder also contains:
    - Code for ambiguity unaware agent.
    - Code for ambiguity unaware agent with backward recursion and cost-to-go computation.
    - Gaussian Process (GP) models and the code to train GP models.
    - Code to reconstruct the cost using trajectory data.  
- Belief Update Benchmark:
  - This folder contains the code files required to replicate the belief update benchmarking results. 
- MaxDiff Benchmark:
  - This folder contains the code files required to replicate the MaxDiff RL benchmarking results on Robotarium and MuJoCo Ant Environments.
- Assets
  - contains all the plots of the manuscript, the data from the experiments used to generate these plots, and the Robotarium movie. 

### Summary of Key Experimental Results
We present the simulation and experimental results given in the manuscript.

*Robotarium:*
- In-silico results:

![Screenshot 2025-03-06 100737](https://github.com/user-attachments/assets/39d90d82-93d9-4a71-be70-41581d8e6679)

Figure. At every training stage, we compare DR-FREE with a free-energy minimizing agent that, while making optimal decisions, does not account for ambiguity. With identical starting positions across experiments, DR-FREE consistently guides the robot to complete its task, whereas the ambiguity-unaware agent fails.

![Screenshot 2025-02-28 140310](https://github.com/user-attachments/assets/60bef038-40c3-4368-b058-5dccbe7e55c5)

Figure. (left panel) The nonconvex state cost for the navigation task. (right panel) Reconstructed cost using the belief updating algorithm.

- Experimental results:

The following videos show robotarium robot performing the task:
  - When the control policy is obtained using DR-FREE Algoritm.


https://github.com/user-attachments/assets/6f488a94-5981-42ad-8888-0a18ce6d943b




### Authors and Contributors 
Author of the code and simulations: *Hozefa Jesawada* (hjesawada@unisa.it)
