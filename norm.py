import pandas as pd
from sklearn.preprocessing import StandardScaler, QuantileTransformer

# Load the dataset
file_path = "./auto_rankings4.parquet"
df = pd.read_parquet(file_path)

# Columns to normalize
ranking_columns = [
    "Contextual_Alignment",
    "Character_Consistency",
    "Descriptive_Depth",
    "Role_Specific_Knowledge",
    "Engagement_and_Collaboration",
    "Creativity_and_Emotional_Nuance",
]

# Z-score normalization
scaler = StandardScaler()
df_zscore = df.copy()
df_zscore[ranking_columns] = scaler.fit_transform(df[ranking_columns])

# Quantile transformation to normal distribution
quantile_transformer = QuantileTransformer(output_distribution="normal", random_state=0)
df_quantile = df.copy()
df_quantile[ranking_columns] = quantile_transformer.fit_transform(df[ranking_columns])

# Save the transformed datasets
zscore_path = "./normalized_zscore.parquet"
quantile_path = "./normalized_quantile.parquet"
df_zscore.to_parquet(zscore_path)
df_quantile.to_parquet(quantile_path)

# info
print("Z-score normalization:")
print(df_zscore[ranking_columns].describe())
print("\nQuantile transformation:")
print(df_quantile[ranking_columns].describe())
# Save the transformed datasets


