import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

def load_and_process_data(file_path):
    """Load pickle file and compute cumulative rewards with zero padding"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    max_length = 1000  # Maximum timesteps
    
    # Pad each episode with zeros to max_length
    padded_data = []
    for episode in data:
        # Create zero array of max_length
        padded_episode = np.zeros(max_length)
        # Fill with actual values
        padded_episode[:len(episode)] = episode
        # Compute cumulative sum
        cumsum_episode = np.cumsum(padded_episode)
        padded_data.append(cumsum_episode)
    
    # Convert to numpy array
    cumsum_data = np.array(padded_data)
    
    # Compute statistics
    mean = np.mean(cumsum_data, axis=0)
    std = np.std(cumsum_data, axis=0)
    
    return mean, std, max_length

def analyze_trajectories(file_path):
    """Analyze trajectory lengths"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Count trajectories shorter than 1000
    lengths = [len(episode) for episode in data]
    short_trajectories = sum(1 for l in lengths if l < 1000)
    
    print(f"\nTrajectory Analysis:")
    print(f"Total trajectories: {len(data)}")
    print(f"Trajectories < 1000 steps: {short_trajectories}")
    print(f"Length statistics:")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {np.mean(lengths):.1f}")
    print(f"  Std: {np.std(lengths):.1f}")
    
    return lengths



# Set style parameters
plt.rcParams.update({
    'font.size': 24,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
})

# Create figure
plt.figure(figsize=(8, 6))

# Load and plot data for each method
methods = {
    r'D:\DR_FREE_ICRA\code\ant_drfree_rewards.pkl': ('DR-FREE', 'blue'),
    r'D:\DR_FREE_ICRA\code\ant_fpd_rewards.pkl': ('Ambiguity Unaware', 'purple'),
    r'D:\DR_FREE_ICRA\code\ant_maxdiff_rewards.pkl': ('MaxDiff', 'red'),
    r'D:\DR_FREE_ICRA\code\ant_mppi_rewards.pkl': ('NN-MPPI', 'green')
}

for file_name, (label, color) in methods.items():
    mean, std, length = load_and_process_data(file_name)
    steps = np.arange(length)
    
    # Add method-specific data processing
    if label == 'DR-FREE':
        mean[:]  # Adjust the value as needed
        std = std   # Scale the std if needed
    elif label == 'Ambiguity Unaware':
        mean[:]  # Adjust the value as needed
        std = std    # Scale the std if needed    
    elif label == 'MaxDiff':
        mean[:]  # Adjust the value as needed
        std = std    # Scale the std if needed
    elif label == 'NN-MPPI':
        mean[:]   # Adjust the value as needed
        std = std    # Scale the std if needed
        
    # Add this before plotting
    print("Analyzing trajectories for each method...")
    for file_path, (method_name, _) in methods.items():
        print(f"\n{method_name}:")
        lengths = analyze_trajectories(file_path)
    
    plt.plot(steps, mean, label=label, color=color, linewidth=2)
    plt.fill_between(steps, mean - std, mean + std, 
                    alpha=0.2, color=color)

# Customize plot
plt.xlabel('Timestep', labelpad=10, fontsize=14)
plt.ylabel('Cumulative Reward', labelpad=10, fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left', frameon=True, framealpha=1.0, fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.xlim(1, 1000)


# Save and show plot
plt.tight_layout()
plt.savefig('ant_cumulative_rewards.jpg', dpi=500, bbox_inches='tight')
plt.show()



# Create a second figure for bar plot
plt.figure(figsize=(10,6))

# Compute final returns (last value of cumsum) for each method
final_returns = []
final_stds = []
method_labels = []

for file_name, (label, color) in methods.items():
    mean, std, length = load_and_process_data(file_name)
    
    # Apply the same scaling as in line plot
    if label == 'DR-FREE':
        mean
        std
    elif label == 'Ambiguity Unaware':
        mean
        std
    elif label == 'MaxDiff':
        mean
        std
    elif label == 'NN-MPPI':
        mean
        std
    
    # Store final return (last value) and its std
    final_returns.append(mean[-1])
    final_stds.append(std[-1])
    method_labels.append(label)

# Create bar plot
bars = plt.bar(method_labels, final_returns, yerr=final_stds, 
               capsize=8, alpha=0.5,color=["blue","red","green"])

plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.ylabel("Total Return", labelpad=10, fontsize=14)

# Annotate mean values above bars
for bar, mean in zip(bars, final_returns):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2.8*final_stds[0]/2,
             f"{mean:.0f}", ha='center', va='top', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig('ant_total_returns.jpg', dpi=500, bbox_inches='tight')
plt.show()
print("Bar plot saved as 'ant_total_returns.svg'")