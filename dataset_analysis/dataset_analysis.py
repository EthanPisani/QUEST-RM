# dataset_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# ===========================
# CONFIGURATION: File Paths & Column Info
# ===========================
# Assume the four datasets are stored as Parquet files.
file_paths = {
    "Dataset1": "../datasets/model1/model1_dataset1.parquet",
    "Dataset2": "../datasets/model2/model2_dataset2.parquet",
    "Dataset3": "../datasets/model3/model3_dataset3.parquet",
    "Dataset4": "../datasets/model4/model4_dataset4.parquet",
}

# Define the numeric columns (score columns) for analysis
numeric_cols = [
    "Contextual_Alignment", "Character_Consistency", "Descriptive_Depth",
    "Role_Specific_Knowledge", "Engagement_and_Collaboration", "Creativity_and_Emotional_Nuance"
]

# ===========================
# LOAD DATA
# ===========================
# Load each dataset from its Parquet file into a dictionary of DataFrames.
datasets = {}
for name, path in file_paths.items():
    try:
        df = pd.read_parquet(path)
        datasets[name] = df
        print(f"{name} loaded successfully with {df.shape[0]} rows.")
    except Exception as e:
        print(f"Error loading {name} from {path}: {e}")

# ===========================
# TECHNIQUE 1: SUMMARY STATISTICS
# ===========================
print("\n--- SUMMARY STATISTICS FOR NUMERIC COLUMNS ---")
for name, df in datasets.items():
    print(f"\n{name} Summary Statistics:")
    print(df[numeric_cols].describe())

# ===========================
# TECHNIQUE 2: DISTRIBUTION VISUALIZATIONS
# ===========================
# For each dataset, create histograms and boxplots for each numeric column.
for name, df in datasets.items():

    
    # Combined boxplot for all numeric columns in the dataset
    plt.figure(figsize=(10, 6))
    plt.boxplot([df[col].dropna() for col in numeric_cols], labels=numeric_cols)
    plt.title(f"{name} - Boxplot of Numeric Scores")
    plt.ylabel("Score")
    plt.grid(True)
    plt.savefig(f"{name}_boxplot.png")

# ===========================
# TECHNIQUE 2: DISTRIBUTION VISUALIZATIONS - Combined Histograms
# ===========================
# For each dataset, create a single histogram showing all 6 attributes
# ===========================
# TECHNIQUE 2: DISTRIBUTION VISUALIZATIONS - Side-by-side Histograms
# ===========================
# For each dataset, create a single histogram showing all 6 attributes side-by-side
for name, df in datasets.items():
    plt.figure(figsize=(14, 8))
    
    # Create bins from 1 to 10 (assuming scores are in this range)
    bins = np.arange(0, 11) - 0.5  # Center bins on integer values
    
    # Plot histograms for each numeric column
    hist_data = []
    for col in numeric_cols:
        hist, _ = np.histogram(df[col].dropna(), bins=bins)
        hist_data.append(hist)
    
    # Convert to numpy array for easier manipulation
    hist_data = np.array(hist_data)
    
    # Set up the bar positions
    bar_width = 0.12  # Width of each bar
    x = np.arange(1, 11)  # Center positions for the groups
    
    # Create bars for each attribute
    for i, col in enumerate(numeric_cols):
        plt.bar(x + (i - 2.5) * bar_width, hist_data[i], width=bar_width, 
                label=col, alpha=0.8)
    
    plt.title(f"{name} - Score Distribution by Attribute")
    plt.xlabel("Score Value")
    plt.ylabel("Frequency")
    plt.xticks(x)  # Set x-ticks to integer score values
    plt.xlim(0.5, 10.5)  # Set x-axis limits to show all scores
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(f"{name}_side_by_side_histogram.png")
    plt.close()
# ===========================
# TECHNIQUE 3: CORRELATION ANALYSIS
# ===========================
# For each dataset, compute and visualize the correlation matrix of numeric columns.
for name, df in datasets.items():
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr, cmap="viridis", interpolation="none")
    plt.colorbar(im)
    plt.xticks(ticks=np.arange(len(numeric_cols)), labels=numeric_cols, rotation=45)
    plt.yticks(ticks=np.arange(len(numeric_cols)), labels=numeric_cols)
    plt.title(f"{name} - Correlation Matrix of Numeric Columns")
    plt.tight_layout()
    plt.savefig(f"{name}_correlation_matrix.png")

# ===========================
# TECHNIQUE 4: TEXTUAL ANALYSIS (TITLE & MESSAGE Lengths)
# ===========================
# Define a helper function to compute text lengths
def compute_text_lengths(df, col):
    print(f"Computing text lengths for column: {col}")
    # Compute both character counts and word counts
    # message col is a list of a dict with key content
    char_counts = df[col].apply(len)
    word_counts = df[col].apply(lambda x: len(str(x).split()))
    return char_counts, word_counts
for name, df in datasets.items():
    for text_col in ["message"]:
        # Ensure non-null values
        if text_col in df.columns:
            char_counts, word_counts = compute_text_lengths(df, text_col)
            
            # Histogram for character count
            plt.figure(figsize=(8, 5))
            plt.hist(char_counts, bins=30, edgecolor="black")
            plt.title(f"{name} - {text_col.capitalize()} Character Count Distribution")
            plt.xlabel("Number of Characters")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.savefig(f"{name}_{text_col}_char_count_distribution.png")
            
            # Histogram for word count
            plt.figure(figsize=(8, 5))
            plt.hist(word_counts, bins=30, edgecolor="black")
            plt.title(f"{name} - {text_col.capitalize()} Word Count Distribution")
            plt.xlabel("Number of Words")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.savefig(f"{name}_{text_col}_word_count_distribution.png")
        else:
            print(f"Column '{text_col}' not found in {name}.")

# ===========================
# TECHNIQUE 5: PRINCIPAL COMPONENT ANALYSIS (PCA)
# ===========================
# Combine the numeric scores from all datasets for a unified PCA analysis.
combined_data = []
dataset_labels = []
for name, df in datasets.items():
    # Select only the numeric columns, drop rows with missing values if any
    subset = df[numeric_cols].dropna().copy()
    subset["Dataset"] = name  # Tag with the dataset name
    combined_data.append(subset)
combined_df = pd.concat(combined_data, ignore_index=True)

# Perform PCA to reduce the six numeric dimensions to two principal components.
pca = PCA(n_components=2)
pca_result = pca.fit_transform(combined_df[numeric_cols])
combined_df["PC1"] = pca_result[:, 0]
combined_df["PC2"] = pca_result[:, 1]
color_map = {
    "Dataset1": "red",
    "Dataset2": "blue",
    "Dataset3": "orange",
    "Dataset4": "green"
}
# Create a color column based on the dataset name

# Plot the PCA results, color-coded by dataset.
plt.figure(figsize=(10, 7))
for name in file_paths.keys():
    subset = combined_df[combined_df["Dataset"] == name]
    plt.scatter(subset["PC1"], subset["PC2"], alpha=0.2, label=name, s=8, color=color_map[name])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Numeric Scores Across Datasets")
plt.legend()
plt.grid(True)
plt.savefig("pca_numeric_scores.png")

# ===========================
# CONCLUSION OF ANALYSES
# ===========================
print("\n--- ANALYSIS COMPLETE ---")
print("1. Summary statistics provide insight into central tendencies and spread across numeric columns.")
print("2. Distribution plots (histograms and boxplots) highlight differences in score distributions between datasets.")
print("3. Correlation matrices reveal the relationships among the different roleplay quality dimensions.")
print("4. Textual analysis of title and message lengths shows content variability across datasets.")
print("5. PCA visualization demonstrates how the datasets cluster based on their numeric score profiles,")
print("   suggesting potential differences in the underlying evaluation criteria.")

# End of script
