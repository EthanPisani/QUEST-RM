import pandas as pd

# Path to the Parquet file
parquet_file = './processed/combined_dataset.parquet'

# Load the dataset from the Parquet file
df = pd.read_parquet(parquet_file)

# Display basic information about the dataset
def explore_dataset(df):
    # Display the first few rows of the dataset
    print("First few rows of the dataset:")
    print(df.head())
    df = df[df['message'].apply(lambda x: len(x) == 2)]
    # Show general information (columns, non-null counts, data types)
    print("\nDataset info:")
    print(df.info())

    # iter over random set of datapoints
    print("\nRandom sample of the dataset:")

    # load processed dataset and remove all rows in that dataset
    processed_df = pd.read_parquet('./processed/sample_dataset.parquet')

    # create new df that does not contain any rows from processed dataset
    df = df[~df.index.isin(processed_df.index)]
    print(df.info())

    # sample 1 million rows
    sample_df = df.sample(1000000)

    # save to new file for processing 2
    sample_df.to_parquet('./processed/sample_dataset_3.parquet', index=True)


   # print(df.sample(5))

    # sampl,e 50k and save to separate file
 #   sample_df = df.sample(50000)
  #  sample_df.to_parquet('./processed/sample_dataset.parquet', index=True)

if __name__ == "__main__":
    explore_dataset(df)
