import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau
import argparse
args = argparse.ArgumentParser()
args.add_argument('--other_rankings', type=str, default="rpbench_rankings.txt", help="Path to the TXT file containing other model rankings")
args.add_argument('--rankings_csv', type=str, default="../model3_g_rankings.csv", help="Path to the CSV file containing model rankings")
args = args.parse_args()
# Load the CSV data
df = pd.read_csv(args.rankings_csv)

# Configure plotting style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]

# Define rankings

other_bench_ranking = []
with open(args.other_rankings, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            other_bench_ranking.append(line)
# print(other_bench_ranking)


# Mapping between model names in CSV and rpbench-auto
model_name_mapping = {
    'Gemini 1.5 Pro 001': 'gemini-1.5-pro-001',
    'GPT-4-Turbo (1106 Preview)': 'gpt-4-1106-preview',
    'Llama-3 70B Instruct': 'llama-3-70b-instruct',
    'Qwen 2 72B Instruct': 'qwen-2-72b-instruct',
    'Llama-3.1 70B Instruct': 'llama-3.1-70b-instruct',
    'Llama-3.1 405B Instruct (FP8)': 'llama-3-1-405b-instruct-fp8',
    'Claude-3.5 Sonnet (2024-06-20)': 'claude-3.5-sonnet-20240620',
    'Higgs-Llama-3 70B V1 3.5bpw': 'higgs-llama-3-70b',
    'Llama-3.1 8B Instruct': 'llama3.1-8B-instruct',
    'Yi Large (2024-05-13)': 'yi-large',
    'GPT-4o (2024-05-13)': 'gpt-4o',
    'Mistral Large (2024-02)': 'mistral-large-2402'
}

# Load the CSV data

# Clean column names by stripping whitespace
df.columns = df.columns.str.strip()

# Prepare comparison data
comparison_data = []
attributes = [col for col in df.columns if col not in ['rank', 'Model']]  # All attributes except rank and Model

for model in df['Model']:
    # if model in model_name_mapping:
        # rpbench_name = model_name_mapping[model]
    comparison_data.append({'Model': model, 'rpbench_name': model})
# print(comparison_data)
comparison_df = pd.DataFrame(comparison_data)
full_df = pd.merge(df, comparison_df, on='Model')

# Create your_ranking from the CSV data (sorted by rank)
your_ranking = full_df.sort_values('rank')['rpbench_name'].tolist()
# print(your_ranking)
# update your_ranking and other_bench_ranking to be only overlapping models
your_ranking = [model for model in your_ranking if model in other_bench_ranking]
other_bench_ranking = [model for model in other_bench_ranking if model in your_ranking]
print(your_ranking)
print(other_bench_ranking)
# Convert model names to rankings
rpbench_positions = {model: i+1 for i, model in enumerate(other_bench_ranking)}
your_positions = {model: i+1 for i, model in enumerate(your_ranking)}

# Get lists of rankings
rpbench_ranks = [rpbench_positions[model] for model in your_ranking]
your_ranks = [your_positions[model] for model in your_ranking]

# Compute correlations
spearman_corr, _ = spearmanr(rpbench_ranks, your_ranks)
kendall_corr, _ = kendalltau(rpbench_ranks, your_ranks)

# Create the combined visualization figure
plt.figure(figsize=(18, 12))
plt.suptitle("Comprehensive Model Ranking Analysis", fontsize=16, y=1.02)

# Grid layout for subplots
gs = plt.GridSpec(2, 2)

# 1. Scatter Plot of Rankings (top left)
ax1 = plt.subplot(gs[0, 0])
ax1.scatter(rpbench_ranks, your_ranks, s=100, alpha=0.7)
for i, model in enumerate(your_ranking):
    ax1.text(rpbench_ranks[i]+0.1, your_ranks[i]+0.1, model, fontsize=16)
ax1.plot([1, 12], [1, 12], 'k--', alpha=0.3)
ax1.set_xlabel(args.other_rankings.split("/")[-1].split(".")[0] + " Rank")
ax1.set_ylabel(args.rankings_csv.split("/")[-1].split(".")[0] + " Rank")
ax1.set_title(f"Ranking Comparison {args.other_rankings.split('/')[-1].split('.')[0]} vs {args.rankings_csv.split('/')[-1].split('.')[0]}\nSpearman ρ = {spearman_corr:.2f}, Kendall τ = {kendall_corr:.2f}")
ax1.grid(True)

# 2. Heatmap of Rank Differences (top middle)
ax2 = plt.subplot(gs[0, 1])
rank_diffs = np.array(rpbench_ranks) - np.array(your_ranks)
sns.heatmap(rank_diffs.reshape(-1, 1), annot=True, cmap="coolwarm", fmt="d", ax=ax2)
ax2.set_yticks(np.arange(len(your_ranking)) + 0.5)
ax2.set_yticklabels(your_ranking, rotation=0)
ax2.set_xticks([])
ax2.set_title(f"Rank Differences Heatmap (Positive = {args.rankings_csv.split('/')[-1].split('.')[0]} Ranking is Higher)")

# # 3. Bar Chart of Rank Differences (middle row)
# ax3 = plt.subplot(gs[1, :])
# ax3.barh(your_ranking, rank_diffs, color="red", alpha=0.7)
# ax3.axvline(0, color="black", linestyle="--")
# ax3.set_xlabel("Rank Difference (RPBench Rank - Your Rank)")
# ax3.set_title("Rank Differences Bar Chart")

# # 4. Attribute Correlation with rpbench Rankings (bottom left)
# ax4 = plt.subplot(gs[2, 0])
# correlations = []
# for attr in attributes:
#     rho, _ = spearmanr(full_df[attr], full_df['rank'])
#     correlations.append({'Attribute': attr, 'Correlation': -rho})
# corr_df = pd.DataFrame(correlations)
# sns.barplot(data=corr_df.sort_values('Correlation', ascending=False), x='Attribute', y='Correlation', palette='viridis', ax=ax4, hue='Attribute',legend=False)
# ax4.axhline(0, color='black', linestyle='--')
# ax4.set_xticks(np.arange(len(attributes)))
# ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
# ax4.set_title("Attribute Correlation with Your Rankings")

# # 5. Heatmap of Attribute Correlations (bottom middle/right)
# ax5 = plt.subplot(gs[2, 1:])
# heatmap_data = full_df[attributes + ['rank']].corr(method='spearman')
# mask = np.triu(np.ones_like(heatmap_data, dtype=bool))
# sns.heatmap(heatmap_data, mask=mask, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, ax=ax5)
# ax5.set_title("Spearman Correlation Heatmap Between Attributes and Rankings")

plt.tight_layout()
plt.savefig(f'charts/ranking_analysis_{args.other_rankings}_{args.rankings_csv.split("/")[-1].split(".")[0]}.png', dpi=300)

# Print correlation results
print(f"\nRanking Correlation Results:")
print(f"Spearman Correlation: {spearman_corr:.3f}")
print(f"Kendall Tau Correlation: {kendall_corr:.3f}")

# Print attribute correlation summary
print("\nAttribute Correlation Summary:")
# print(corr_df.sort_values('Correlation', ascending=False).to_string(index=False))