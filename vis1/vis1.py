import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# Configure plotting style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]

# rpbench-auto rankings
rpbench_rankings = {
    'claude-3.5-sonnet-20240620': 1,
    'llama-3-1-405b-instruct-fp8': 2,
    'gpt-4-1106-preview': 3,
    'llama-3.1-70b-instruct': 4,
    'gpt-4o': 5,
    'yi-large': 6,
    'higgs-llama-3-70b': 7,
    'qwen-2-72b-instruct': 8,
    'llama-3-70b-instruct': 9,
    'llama3.1-8B-instruct': 10,
    'gemini-1.5-pro': 11,
    'mistral-large-2402': 12
}

# Mapping between model names in CSV and rpbench-auto
model_name_mapping = {
    'Gemini 1.5 Pro 001': 'gemini-1.5-pro',
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
df = pd.read_csv("model3_g_rankings.csv")

# Clean column names by stripping whitespace
df.columns = df.columns.str.strip()

# Prepare comparison data
comparison_data = []
attributes = [col for col in df.columns if col not in ['rank', 'Model']]  # All attributes except rank and Model

for model in df['Model']:
    if model in model_name_mapping:
        rpbench_name = model_name_mapping[model]
        rpbench_rank = rpbench_rankings[rpbench_name]
        comparison_data.append({'Model': model, 'rpbench_rank': rpbench_rank})

comparison_df = pd.DataFrame(comparison_data)
full_df = pd.merge(df, comparison_df, on='Model')

# Calculate correlations and prepare data for visualization
correlations = []
for attr in attributes:
    # Spearman correlation for rank correlation
    rho, p = spearmanr(full_df[attr], full_df['rpbench_rank'])
    correlations.append({'Attribute': attr, 'Correlation': -rho, 'Absolute': abs(rho)})  # Negative because higher scores should correlate with lower (better) ranks

corr_df = pd.DataFrame(correlations)

# Visualization 1: Correlation Heatmap
plt.figure(figsize=(10, 8))
heatmap_data = full_df[attributes + ['rpbench_rank']].corr(method='spearman')
mask = np.triu(np.ones_like(heatmap_data, dtype=bool))
sns.heatmap(heatmap_data, mask=mask, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title("Spearman Correlation Heatmap Between Attributes and rpbench Rankings")
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

# Visualization 2: Attribute Correlation with rpbench Rankings
plt.figure(figsize=(12, 6))
sns.barplot(data=corr_df.sort_values('Correlation', ascending=False), x='Attribute', y='Correlation', palette='viridis')
plt.xticks(rotation=45, ha='right')
plt.title("Attribute Correlation with rpbench Rankings (Higher = Better Alignment)")
plt.axhline(0, color='black', linestyle='--')
plt.tight_layout()
plt.savefig('attribute_correlation.png')
plt.show()

# Visualization 3: Weight of Each Attribute in Original Ranking
attribute_weights = df[attributes].std().sort_values(ascending=False)  # Using standard deviation as proxy for weight
plt.figure(figsize=(12, 6))
sns.barplot(x=attribute_weights.index, y=attribute_weights.values, palette='rocket')
plt.xticks(rotation=45, ha='right')
plt.title("Relative Weight of Each Attribute (Standard Deviation)")
plt.ylabel("Standard Deviation")
plt.tight_layout()
plt.savefig('attribute_weights.png')
plt.show()

# Print summary statistics
print("\nAttribute Correlation Summary:")
print(corr_df.sort_values('Correlation', ascending=False).to_string(index=False))

print("\nAttribute Weight Summary:")
print(attribute_weights.to_string())