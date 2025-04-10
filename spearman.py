import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau

# Define rankings
rpbench_ranking = [
    "claude-3.5-sonnet-20240620", "llama-3-1-405b-instruct-fp8", "gpt-4-1106-preview",
    "llama-3.1-70b-instruct", "gpt-4o", "yi-large", "higgs-llama-3-70b",
    "qwen-2-72b-instruct", "llama-3-70b-instruct", "llama3.1-8B-instruct",
    "gemini-1.5-pro-001", "mistral-large-2402"
]



your_ranking = [
    "gemini-1.5-pro-001", "gpt-4-1106-preview", "llama-3-70b-instruct",
    "qwen-2-72b-instruct", "llama-3.1-70b-instruct", "llama-3-1-405b-instruct-fp8",
    "claude-3.5-sonnet-20240620", "higgs-llama-3-70b", "llama3.1-8B-instruct",
    "yi-large", "gpt-4o", "mistral-large-2402"
]
# Convert model names to rankings
rpbench_positions = {model: i+1 for i, model in enumerate(rpbench_ranking)}
your_positions = {model: i+1 for i, model in enumerate(your_ranking)}

# Get lists of rankings
rpbench_ranks = [rpbench_positions[model] for model in your_ranking]
your_ranks = [your_positions[model] for model in your_ranking]

# Compute correlations
spearman_corr, _ = spearmanr(rpbench_ranks, your_ranks)
kendall_corr, _ = kendalltau(rpbench_ranks, your_ranks)

# Create 3 graphs
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Scatter Plot of Rankings
axes[0].scatter(rpbench_ranks, your_ranks)
axes[0].set_xlabel("RPBench Rank")
axes[0].set_ylabel("Your Model Rank")
axes[0].set_title("Ranking Comparison Scatter Plot")
axes[0].grid(True)

# 2. Heatmap of Rank Differences
rank_diffs = np.array(rpbench_ranks) - np.array(your_ranks)
sns.heatmap(rank_diffs.reshape(-1, 1), annot=True, cmap="coolwarm", fmt="d", ax=axes[1])
axes[1].set_yticks(np.arange(len(your_ranking)) + 0.5)
axes[1].set_yticklabels(your_ranking, rotation=0)
axes[1].set_xticks([])
axes[1].set_title("Rank Differences Heatmap")

# 3. Bar Chart of Rank Differences
axes[2].barh(your_ranking, rank_diffs, color="red", alpha=0.7)
axes[2].axvline(0, color="black", linestyle="--")
axes[2].set_xlabel("Rank Difference (Your Rank - RPBench Rank)")
axes[2].set_title("Rank Differences Bar Chart")

# Show the plots
plt.tight_layout()
plt.savefig("rankings_comparison.png")

# Output correlation results
print(f"Spearman Correlation: {spearman_corr:.2f}")
print(f"Kendall Tau Correlation: {kendall_corr:.2f}")
