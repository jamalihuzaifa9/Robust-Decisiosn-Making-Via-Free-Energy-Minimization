# Re-import necessary libraries due to kernel reset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

# Provided results dictionary
results = {
    (2, 10): 3, (2, 20): 4, (2, 30): 4, (2, 50): 3, (2, 100): 4, (2, 200): 4,
    (5, 10): 6, (5, 20): 5, (5, 30): 4, (5, 50): 5, (5, 100): 5, (5, 200): 6,
    (10, 10): 8, (10, 20): 7, (10, 30): 7, (10, 50): 8, (10, 100): 9, (10, 200): 9,
    (20, 10): 8, (20, 20): 11, (20, 30): 11, (20, 50): 12, (20, 100): 12, (20, 200): 12,
    (30, 10): 7, (30, 20): 10, (30, 30): 12, (30, 50): 12, (30, 100): 12, (30, 200): 12,
    (50, 10): 3, (50, 20): 3, (50, 30): 5, (50, 50): 7, (50, 100): 10, (50, 200): 11
}

# Grid search parameters
horizon_list = [2, 5, 10, 20, 30, 50]
samples_list = [10, 20, 30, 50, 100, 200]

# Initialize the heatmap matrix with NaNs
heatmap_matrix = np.full((len(horizon_list), len(samples_list)), np.nan)

# Fill known results
for (h, s), val in results.items():
    i = horizon_list.index(h)
    j = samples_list.index(s)
    heatmap_matrix[i, j] = val

# Convert success counts to percentage (out of 12)
heatmap_percentage = (heatmap_matrix / 12.0) * 100

# Transpose for horizon on x-axis and samples on y-axis
heatmap_percentage_T = heatmap_percentage.T

# Plot the heatmap with percentage annotations
plt.figure(figsize=(8, 6))
ax = sns.heatmap(
    heatmap_percentage_T,
    annot=False,
    fmt=".0f",
    xticklabels=horizon_list,
    yticklabels=samples_list,
    cmap="YlGnBu",
    linewidths=0.5,
    linecolor='gray',
    cbar=True,
    mask=np.isnan(heatmap_percentage_T)
)

# Set font size for axis labels
plt.xlabel("Horizon", fontsize=14)
plt.ylabel("Samples", fontsize=14)

# Set font size for tick labels
ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)

ax.invert_yaxis()
plt.tight_layout()
# plt.show()
plt.savefig("grid_search_heatmap.jpg", dpi=500, bbox_inches='tight')
