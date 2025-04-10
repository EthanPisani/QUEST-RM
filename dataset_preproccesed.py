import os
import pandas as pd
import json
import tqdm
import multiprocessing
import re

# Dataset configurations
datasets = [
    {
        "path": '/mnt/m2kingston/dev/datasets/FIREBALL',
        "name": 'FIREBALL',
        "type": 'jsonl'
    },
    {
        "path": '/mnt/m2kingston/dev/datasets/CharacterChronicles/roleplayerguild',
        "name": 'roleplayerguild',
        "type": 'csv'
    },
    {
        "path": '/mnt/m2kingston/dev/datasets/CharacterChronicles/roleplay-by-post',
        "name": 'roleplay-by-post',
        "type": 'csv'
    },
    {
        "path": '/mnt/m2kingston/dev/datasets/CharacterChronicles/giantinplayground',
        "name": 'giantplayground',
        "type": 'csv'
    },
    {
        "path": '/mnt/m2kingston/dev/datasets/CharacterChronicles/Elliquiy',
        "name": 'Elliquiy',
        "type": 'csv'
    }
]

# HTML cleaner
def clean_html(raw_text):
    raw_text = str(raw_text)
    clean_text = re.sub(r'<.*?>', ' ', raw_text)
    clean_text = re.sub(r'&\w+;', ' ', clean_text)
    return clean_text

# Preprocess CSV datasets
def preprocess_dataset_character_chronicles(csv_file,dataset_name):
    print(f"Reading csv file {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"Preprocessing dataset from {csv_file}")

    grouped = df.groupby('thread_title', sort=False)
    formatted_data = []
    for thread_title, group in tqdm.tqdm(grouped):
        if "OOC" in thread_title:
            print(f"Skipping OOC thread: {thread_title}")
            continue
        group = group.sort_values(by='message_timestamp')
        messages = group['message'].tolist()

        for i in range(len(messages) - 1):
            user_message = {"content": clean_html(messages[i]), "role": "user"}
            assistant_message = {"content": clean_html(messages[i + 1]), "role": "assistant"}
            if not user_message['content'] or not assistant_message['content']:
                continue
            formatted_entry = {
                "dataset": dataset_name,
                "title": thread_title,
                "message": json.dumps([user_message, assistant_message])
            }
            formatted_data.append(formatted_entry)

    return formatted_data

def preprocess_thread_character_chronicles(csv_file, dataset_name):
    return preprocess_dataset_character_chronicles(csv_file, dataset_name)

# Process CSV files for each dataset
def process_dataset(dataset_index):
    csv_files = []
    path = datasets[dataset_index]['path']
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    all_data = []
    for csv_file in csv_files:
        all_data.extend(preprocess_thread_character_chronicles(csv_file, datasets[dataset_index]['name']))

    return all_data

# Combine data into a Parquet file
def combine_to_parquet(all_data, output_parquet_file):
    df = pd.json_normalize(all_data)
    df.to_parquet(output_parquet_file, index=True)
    print(f"Combined dataset saved to {output_parquet_file}")

# Main execution
def main():
    all_data = []
    all_data.extend(process_dataset(2))  # roleplay-by-post
    all_data.extend(process_dataset(1))  # roleplayerguild
    all_data.extend(process_dataset(3))  # giantplayground
    all_data.extend(process_dataset(4))  # Elliquiy

    # Combine all processed data into a Parquet file
    combine_to_parquet(all_data, './processed/combined_dataset.parquet')

if __name__ == "__main__":
    main()
