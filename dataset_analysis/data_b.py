import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

# ------------------------------
# CONFIGURATION & DATA LOADING
# ------------------------------
# Update this file path as needed.

file_paths = {
    "Dataset1": "../datasets/model1/auto_rankings1.parquet",
    "Dataset2": "../datasets/model2/auto_rankings2.parquet",
    "Dataset3": "../datasets/model3/auto_rankings3.parquet",
    "Dataset4": "../datasets/model4/auto_rankings4.parquet"
}

# Assume the CSV contains a column 'dataset' indicating which dataset each record comes from.
# Define the six score columns (from the provided dataset description)
score_cols = [
    "Contextual_Alignment", "Character_Consistency", "Descriptive_Depth",
    "Role_Specific_Knowledge", "Engagement_and_Collaboration", "Creativity_and_Emotional_Nuance"
]

# create 1 dataset with dataset column
df = pd.DataFrame()
for name, path in file_paths.items():
    try:
        temp_df = pd.read_parquet(path)
        temp_df['dataset'] = name  # Add a column to indicate the dataset
        df = pd.concat([df, temp_df], ignore_index=True)
        print(f"{name} loaded successfully with {temp_df.shape[0]} rows.")
    except Exception as e:
        print(f"Error loading {name} from {path}: {e}")

# ------------------------------
# 1. SUMMARY STATISTICS & LEADERBOARD
# ------------------------------
print("Summary Statistics for Score Columns:\n")
print(df[score_cols].describe())

# Create a leaderboard by sorting models by one key metric (if available).
if 'Average Score' in df.columns:
    leaderboard = df.sort_values("Average Score", ascending=False)[["model", "Average Score"]]
    print("\nLeaderboard (by Average Score):")
    print(leaderboard.to_string(index=False))
else:
    # Otherwise, sort by mean of score_cols for each model.
    df['Mean_Score'] = df[score_cols].mean(axis=1)
    leaderboard = df.sort_values("Mean_Score", ascending=False)[["model", "Mean_Score"]]
    print("\nLeaderboard (by Mean Score of all dimensions):")
    print(leaderboard.to_string(index=False))

# ------------------------------
# 2. LAYERED HISTOGRAMS FOR EACH DATASET
# ------------------------------
# If the dataframe contains a 'dataset' column, group by it.
if "dataset" in df.columns:
    datasets = df["dataset"].unique()
    for d in datasets:
        df_d = df[df["dataset"] == d]
        plt.figure(figsize=(10, 6))
        bins = 20  # number of bins for histograms
        colors = sns.color_palette("husl", len(score_cols))
        
        # Plot each score column's histogram on the same axes.
        for idx, col in enumerate(score_cols):
            # Plot using density=True to compare distributions on the same scale.
            plt.hist(df_d[col].dropna(), bins=bins, alpha=0.5, density=True,
                     color=colors[idx], label=col)
        
        plt.title(f"Layered Histograms of Score Dimensions for Dataset: {d}")
        plt.xlabel("Score Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"layered_histograms_{d}.png", dpi=300)
else:
    print("No 'dataset' column found. Plotting layered histograms for the entire data.")
    plt.figure(figsize=(10, 6))
    bins = 20
    colors = sns.color_palette("husl", len(score_cols))
    for idx, col in enumerate(score_cols):
        plt.hist(df[col].dropna(), bins=bins, alpha=0.5, density=True,
                 color=colors[idx], label=col)
    plt.title("Layered Histograms of Score Dimensions for All Data")
    plt.xlabel("Score Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("layered_histograms_all_data.png", dpi=300)

# ------------------------------
# 3. CORRELATION ANALYSIS
# ------------------------------
corr_matrix = df[score_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix of Score Dimensions")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=300)
# ------------------------------
# 4. SCATTER PLOTS: COMPARING DIMENSIONS
# ------------------------------
# Example scatter plot: Contextual_Alignment vs. Character_Consistency
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Contextual_Alignment", y="Character_Consistency", hue="model", alpha=0.7)
plt.title("Contextual Alignment vs. Character Consistency")
plt.xlabel("Contextual Alignment")
plt.ylabel("Character Consistency")
plt.grid(True)
plt.tight_layout()
plt.savefig("scatter_contextual_vs_character.png", dpi=300)

# ------------------------------
# 5. PRINCIPAL COMPONENT ANALYSIS (PCA)
# ------------------------------
pca = PCA(n_components=2)
pca_data = pca.fit_transform(df[score_cols].dropna())
df_pca = pd.DataFrame(data=pca_data, columns=["PC1", "PC2"])
df_pca["model"] = df["model"]

plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="model", palette="tab10", s=100, alpha=0.8)
plt.title("PCA of Score Dimensions")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_score_dimensions.png", dpi=300)

# ------------------------------
# 6. OPTIONAL REGRESSION ANALYSIS
# ------------------------------
import statsmodels.api as sm

if "Contextual_Alignment" in df.columns and "Average Score" in df.columns:
    X = df["Contextual_Alignment"]
    y = df["Average Score"]
    X = sm.add_constant(X)
    reg_model = sm.OLS(y, X).fit()
    print("\nRegression Analysis: Predicting Average Score from Contextual Alignment")
    print(reg_model.summary())
else:
    print("Skipping regression analysis since required columns are not available.")

print("\nAnalysis Complete. The layered histograms, correlation matrices, scatter plots, PCA, and optional regression provide a multi-faceted view of reward model performance across datasets.")
