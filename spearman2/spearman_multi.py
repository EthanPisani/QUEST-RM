import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau
import argparse
import glob
import os

# Set up argument parser
parser = argparse.ArgumentParser(description='Compare multiple ranking files using correlation metrics.')
parser.add_argument('--ranking_files', type=str, nargs='+', required=True,
                    help='List of ranking files to compare (txt or csv)')
parser.add_argument('--output', type=str, default='ranking_comparison.png',
                    help='Output filename for the plot')
args = parser.parse_args()

# Configure plotting style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]

def load_ranking(file_path):
    """Load a ranking from either txt or csv file"""
    if file_path.endswith('.txt'):
        ranking = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    ranking.append(line)
        return ranking
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        if 'Model' in df.columns and 'rank' in df.columns:
            return df.sort_values('rank')['Model'].tolist()
        elif 'Model' in df.columns:
            return df['Model'].tolist()
        else:
            return df.iloc[:, 0].tolist()  # Assume first column is model names
    else:
        raise ValueError("Unsupported file format. Please provide .txt or .csv files")

# Load all rankings
rankings = {}
for file_path in args.ranking_files:
    name = os.path.splitext(os.path.basename(file_path))[0]
    rankings[name] = load_ranking(file_path)

# Get all unique model names
all_models = set()
for ranking in rankings.values():
    all_models.update(ranking)
all_models = sorted(all_models)

# Create a dictionary to store each ranking's positions for all models
# Models not in a ranking will have NaN position
ranking_positions = {}
for name, ranking in rankings.items():
    positions = {model: i+1 for i, model in enumerate(ranking)}
    # For models not in this ranking, we'll use NaN
    ranking_positions[name] = [positions.get(model, np.nan) for model in all_models]

# Calculate correlation matrix using pairwise common models
names = list(rankings.keys())
n = len(names)
spearman_matrix = np.zeros((n, n))
kendall_matrix = np.zeros((n, n))
common_counts = np.zeros((n, n), dtype=int)

for i in range(n):
    for j in range(n):
        if i == j:
            spearman_matrix[i,j] = np.nan
            kendall_matrix[i,j] = np.nan
            common_counts[i,j] = 0
        else:
            # Get positions for this pair of rankings
            r1 = ranking_positions[names[i]]
            r2 = ranking_positions[names[j]]
            
            # Find models present in both rankings
            mask = ~np.isnan(r1) & ~np.isnan(r2)
            common_models = np.array(all_models)[mask]
            r1_filtered = np.array(r1)[mask]
            r2_filtered = np.array(r2)[mask]
            
            common_counts[i,j] = len(common_models)
            
            if len(common_models) >= 2:  # Need at least 2 elements to compute correlation
                spearman_matrix[i,j], _ = spearmanr(r1_filtered, r2_filtered)
                kendall_matrix[i,j], _ = kendalltau(r1_filtered, r2_filtered)
            else:
                spearman_matrix[i,j] = np.nan
                kendall_matrix[i,j] = np.nan

# remove higher triangle of the matrix
# mask = np.triu(np.ones_like(spearman_matrix, dtype=bool), k=1)
# spearman_matrix = np.where(mask, np.nan, spearman_matrix)
# kendall_matrix = np.where(mask, np.nan, kendall_matrix)
# Set diagonal to NaN

# Update the heatmap creation part to show both correlation and count
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Ranking Comparison Correlation Matrices (Spearman and Kendall)', fontsize=16)

# Function to create combined annotations
def make_annotations(matrix, counts):
    annotations = np.empty_like(matrix, dtype=object)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == j:
                annotations[i,j] = ""
            elif np.isnan(matrix[i,j]):
                annotations[i,j] = "N/A"
            else:
                annotations[i,j] = f"{matrix[i,j]:.2f}\nn={counts[i,j]}"
    return annotations

# Create combined annotations
spearman_annot = make_annotations(spearman_matrix, common_counts)
kendall_annot = make_annotations(kendall_matrix, common_counts)

# Plot Spearman correlation with combined annotation
sns.heatmap(spearman_matrix, annot=spearman_annot, fmt="", cmap="coolwarm", 
            center=0, vmin=-1, vmax=1, ax=ax1,
            xticklabels=names, yticklabels=names)
ax1.set_title('Spearman Rank Correlation')
ax1.tick_params(axis='x', rotation=45)
ax1.tick_params(axis='y', rotation=0)

# Plot Kendall correlation with combined annotation
sns.heatmap(kendall_matrix, annot=kendall_annot, fmt="", cmap="coolwarm", 
            center=0, vmin=-1, vmax=1, ax=ax2,
            xticklabels=names, yticklabels=names)
ax2.set_title('Kendall Tau Correlation')
ax2.tick_params(axis='x', rotation=45)
ax2.tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig(args.output, dpi=300, bbox_inches='tight')
print(f"Plot saved to {args.output}")

# Print correlation matrices
print("\nSpearman Rank Correlation Matrix:")
print(pd.DataFrame(spearman_matrix, index=names, columns=names))

print("\nKendall Tau Correlation Matrix:")
print(pd.DataFrame(kendall_matrix, index=names, columns=names))

print("\nCommon Model Counts Matrix:")
print(pd.DataFrame(common_counts, index=names, columns=names))