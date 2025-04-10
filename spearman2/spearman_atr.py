import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib
import argparse
args = argparse.ArgumentParser()
args.add_argument('--other_rankings', type=str, default="rpbench.txt", help="Path to the TXT file containing other model rankings")
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


# Calculate correlations for each attribute
results = []
for attr in attributes:
    # Get the rankings for the current attribute
    attr_ranking = full_df.sort_values(attr)['rpbench_name'].tolist()
    # Filter to only include models present in both rankings
    filtered_rpbench_ranking = [model for model in attr_ranking if model in your_ranking]
    filtered_your_ranking = [model for model in your_ranking if model in attr_ranking]
    # Convert model names to rankings
    rpbench_positions = {model: i+1 for i, model in enumerate(filtered_rpbench_ranking)}
    your_positions = {model: i+1 for i, model in enumerate(filtered_your_ranking)}
    sorted_models = [model for model in filtered_rpbench_ranking if model in your_ranking]
    rpbench_ranks = [rpbench_positions[model] for model in sorted_models]
    attr_ranks = sorted_models
    
    spearman, _ = spearmanr(rpbench_ranks, attr_ranks)
    kendall, _ = kendalltau(rpbench_ranks, attr_ranks)
    results.append({
        'Attribute': attr,
        'Spearman': spearman,
        'Kendall': kendall,
        'Models': ", ".join(sorted_models)
    })

results_df = pd.DataFrame(results)

# Create color palette
palette = sns.color_palette("husl", len(attributes))
color_dict = {attr: palette[i] for i, attr in enumerate(attributes)}

# Create scatter plot
plt.figure(figsize=(14, 8))
scatter = sns.scatterplot(
    data=results_df,
    x='Spearman',
    y='Kendall',
    hue='Attribute',
    palette=color_dict,
    s=150,
    alpha=0.8
)

# Add model names as annotations
for i, row in results_df.iterrows():
    plt.annotate(
        row['Attribute'],
        (row['Spearman'], row['Kendall']),
        textcoords="offset points",
        xytext=(0,10),
        ha='center',
        fontsize=9
    )

# Add reference lines
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.axline((0, 0), slope=1, color='gray', linestyle=':', alpha=0.5)

# Customize plot
plt.title('Model Ranking Correlations by Attribute', pad=20)
plt.xlabel('Spearman Correlation (ρ)')
plt.ylabel('Kendall Tau Correlation (τ)')
plt.xlim(min(results_df['Spearman']) - 0.1, max(results_df['Spearman']) + 0.1)
plt.ylim(min(results_df['Kendall']) - 0.1, max(results_df['Kendall']) + 0.1)

# Create custom legend
handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[attr], 
            markersize=10, label=attr) for attr in attributes]
plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(f'attribute_correlation_scatter_{args.rankings_csv.split("/")[-1].split(".")[0]}.png', dpi=300, bbox_inches='tight')
plt.show()

# Print correlation table
print("Attribute Correlation Results:")
print(results_df[['Attribute', 'Spearman', 'Kendall']].sort_values('Spearman', ascending=False).to_string(index=False))