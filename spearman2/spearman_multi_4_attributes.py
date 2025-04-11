# spearman_multi_4_attributes.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau
import argparse
import os

# Set up argument parser
parser = argparse.ArgumentParser(description='Compare multiple ranking files using correlation and ranking similarity metrics.')
parser.add_argument('--ranking_files', type=str, nargs='+', required=True,
                    help='List of ranking files to compare (txt or csv)')
parser.add_argument('--output', type=str, default='ranking_comparison_attributes.png',
                    help='Output filename for the plot')
args = parser.parse_args()

# Configure plotting style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [18, 14]

def load_ranking(file_path):
    """Load a ranking from either a txt or csv file"""
    if file_path.endswith('.txt'):
        ranking = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    ranking.append(line)
        return {'text_ranking': ranking}
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        rankings = {}
        
        # Get all numeric columns (excluding the model name column)
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        model_col = None
        
        # Try to find the model name column
        possible_model_cols = ['Model', 'model', 'Name', 'name', 'System', 'system']
        for col in possible_model_cols:
            if col in df.columns:
                model_col = col
                break
        if model_col is None:
            model_col = df.columns[0]  # Fallback to first column
            
        # Create rankings for each numeric column
        for col in numeric_cols:
            # Sort by the column (ascending for ranks, descending for scores)
            # We assume higher values are better (like accuracy), so we sort descending
            sorted_df = df.sort_values(by=col, ascending=False)
            rankings[col] = sorted_df[model_col].tolist()
            
        return rankings
    else:
        raise ValueError("Unsupported file format. Please provide .txt or .csv files")

def rbo_score(S, T, p=0.9):
    """
    Compute an approximation of Rank-Biased Overlap (RBO) between two ranked lists S and T.
    This implementation iteratively calculates the weighted overlap at each depth.
    """
    k = max(len(S), len(T))
    S_set, T_set = set(), set()
    summation = 0.0
    for d in range(1, k + 1):
        if d <= len(S):
            S_set.add(S[d-1])
        if d <= len(T):
            T_set.add(T[d-1])
        overlap = len(S_set.intersection(T_set))
        summation += (overlap / d) * (p ** (d - 1))
    return (1 - p) * summation

# Load all rankings
rankings = {}
for file_path in args.ranking_files:
    name = os.path.splitext(os.path.basename(file_path))[0]
    loaded = load_ranking(file_path)
    
    if file_path.endswith('.txt'):
        rankings[name] = loaded['text_ranking']
    else:
        # For CSV files, add each column's ranking with a combined name
        for col_name, ranking in loaded.items():
            # dont use rank col
            if col_name in ['rank', 'Model']:
                continue
            rankings[f"{col_name[:30]}"] = ranking

# Get all unique model names (union of all rankings)
all_models = set()
for ranking in rankings.values():
    all_models.update(ranking)
all_models = sorted(all_models)

# Create a dictionary to store each ranking's positions for all models.
# For models not in a ranking, we assign NaN.
ranking_positions = {}
for name, ranking in rankings.items():
    positions = {model: i+1 for i, model in enumerate(ranking)}
    ranking_positions[name] = [positions.get(model, np.nan) for model in all_models]

# Prepare to calculate pairwise comparisons
names = list(rankings.keys())
n = len(names)

# Initialize matrices for Spearman, Kendall, MAR, and RBO, and count matrix for common models
spearman_matrix = np.zeros((n, n))
kendall_matrix = np.zeros((n, n))
mar_matrix = np.zeros((n, n))
rbo_matrix = np.zeros((n, n))
common_counts = np.zeros((n, n), dtype=int)

for i in range(n):
    for j in range(n):
        if i == j:
            spearman_matrix[i, j] = np.nan
            kendall_matrix[i, j] = np.nan
            mar_matrix[i, j] = np.nan
            rbo_matrix[i, j] = np.nan
            common_counts[i, j] = 0
        else:
            # Get positions for the two rankings
            r1 = ranking_positions[names[i]]
            r2 = ranking_positions[names[j]]
            
            # Find models present in both rankings
            mask = ~np.isnan(r1) & ~np.isnan(r2)
            common_counts[i, j] = np.sum(mask)
            r1_filtered = np.array(r1)[mask]
            r2_filtered = np.array(r2)[mask]
            
            # Compute Spearman and Kendall correlations if possible
            if len(r1_filtered) >= 2:
                spearman_matrix[i, j], _ = spearmanr(r1_filtered, r2_filtered)
                kendall_matrix[i, j], _ = kendalltau(r1_filtered, r2_filtered)
            else:
                spearman_matrix[i, j] = np.nan
                kendall_matrix[i, j] = np.nan
            
            # Compute Mean Absolute Rank Difference (MAR) if there is at least one common model
            if len(r1_filtered) > 0:
                mar_matrix[i, j] = np.mean(np.abs(r1_filtered - r2_filtered))
            else:
                mar_matrix[i, j] = np.nan
            
            # Compute RBO using the full raw ranking lists
            rbo_matrix[i, j] = rbo_score(rankings[names[i]], rankings[names[j]])

# Function to create annotations for matrices that include common counts
def make_annotations(matrix, counts, fmt="{:.2f}", skip_nan=True):#\nn={}
    annotations = np.empty_like(matrix, dtype=object)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == j:
                annotations[i, j] = ""
            elif np.isnan(matrix[i, j]) and skip_nan:
                annotations[i, j] = "N/A"
            else:
                annotations[i, j] = fmt.format(matrix[i, j])# ,counts[i, j]
    return annotations

# Create annotations for Spearman, Kendall, and MAR matrices.
spearman_annot = make_annotations(spearman_matrix, common_counts)
kendall_annot = make_annotations(kendall_matrix, common_counts)
mar_annot = make_annotations(mar_matrix, common_counts, fmt="{:.2f}")#\nn={}

# For RBO, we'll annotate with the score only (since it is computed on the full list)
rbo_annot = np.empty_like(rbo_matrix, dtype=object)
for i in range(n):
    for j in range(n):
        if i == j:
            rbo_annot[i, j] = ""
        elif np.isnan(rbo_matrix[i, j]):
            rbo_annot[i, j] = "N/A"
        else:
            rbo_annot[i, j] = f"{rbo_matrix[i, j]:.2f}"

# Create a figure with 2 rows and 2 columns for the four matrices
fig, axs = plt.subplots(2, 2, figsize=(18, 14))
files_names = ', '.join([os.path.basename(file).split(".")[0] for file in args.ranking_files])
fig.suptitle(f'Ranking Comparison: Spearman, Kendall, MAR, and RBO \n{files_names}', fontsize=18)

# Plot Spearman correlation matrix
sns.heatmap(spearman_matrix, annot=spearman_annot, fmt="", cmap="coolwarm", 
            center=0, vmin=-1, vmax=1, ax=axs[0, 0],
            xticklabels=names, yticklabels=names)
axs[0, 0].set_title('Spearman Rank Correlation')
axs[0, 0].tick_params(axis='x', rotation=45)
axs[0, 0].tick_params(axis='y', rotation=0)

# Plot Kendall correlation matrix
sns.heatmap(kendall_matrix, annot=kendall_annot, fmt="", cmap="coolwarm", 
            center=0, vmin=-1, vmax=1, ax=axs[0, 1],
            xticklabels=names, yticklabels=names)
axs[0, 1].set_title('Kendall Tau Correlation')
axs[0, 1].tick_params(axis='x', rotation=45)
axs[0, 1].tick_params(axis='y', rotation=0)

# Plot Mean Absolute Rank Difference (MAR) matrix
sns.heatmap(mar_matrix, annot=mar_annot, fmt="", cmap="YlGnBu", ax=axs[1, 0],
            xticklabels=names, yticklabels=names)
axs[1, 0].set_title('Mean Absolute Rank Difference (MAR)')
axs[1, 0].tick_params(axis='x', rotation=45)
axs[1, 0].tick_params(axis='y', rotation=0)

# Plot Rank-Biased Overlap (RBO) matrix
sns.heatmap(rbo_matrix, annot=rbo_annot, fmt="", cmap="YlOrBr", vmin=0, vmax=1, ax=axs[1, 1],
            xticklabels=names, yticklabels=names)
axs[1, 1].set_title('Rank-Biased Overlap (RBO)')
axs[1, 1].tick_params(axis='x', rotation=45)
axs[1, 1].tick_params(axis='y', rotation=0)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(args.output, dpi=300, bbox_inches='tight')
print(f"Plot saved to {args.output}")

# Print matrices to console
print("\nSpearman Rank Correlation Matrix:")
print(pd.DataFrame(spearman_matrix, index=names, columns=names))

print("\nKendall Tau Correlation Matrix:")
print(pd.DataFrame(kendall_matrix, index=names, columns=names))

print("\nMean Absolute Rank Difference (MAR) Matrix:")
print(pd.DataFrame(mar_matrix, index=names, columns=names))

print("\nRank-Biased Overlap (RBO) Matrix:")
print(pd.DataFrame(rbo_matrix, index=names, columns=names))

print("\nCommon Model Counts Matrix (for rank-based comparisons):")
print(pd.DataFrame(common_counts, index=names, columns=names))