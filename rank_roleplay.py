from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
import os

app = Flask(__name__)
# https://www.reddit.com/r/OriginalCharacter_RP/comments/1b905cq/the_ultimate_guide_on_how_to_roleplay/
# List of features to rank
features = [
    "Contextual Alignment",
    "Character Consistency",
    "Descriptive Depth",
    "Role-Specific Knowledge",
    "Engagement and Collaboration",
    "Creativity and Emotional Nuance"
]

# Path to the Parquet file
parquet_file = "processed/sample_dataset.parquet"

# Load the Parquet file using pandas
def load_roleplays():
    df = pd.read_parquet(parquet_file)
    print(df.head())

    roleplays = []
    for _, row in df.iterrows():
        roleplays.append({
            "id": row.name,  # Use the row index as the unique id
            "dataset": row['dataset'],
            "title": row['title'],
            "message": json.loads(row['message'])  # Assuming 'message' column is stored as a JSON string
        })
    return roleplays

# Dummy placeholder for roleplays, replaced by loaded data
roleplays = load_roleplays()

# Path to save rankings (this will be updated in the Parquet file)
rankings_parquet_file = "rankings.parquet"
rankings_csv_file = "rankings.csv"
if not os.path.exists(rankings_csv_file):
    # Create an empty DataFrame for rankings if the file does not exist
    pd.DataFrame(columns=["roleplay_id", "dataset", "title", "message", *features]).to_csv(rankings_csv_file, index=False)
    
if not os.path.exists(rankings_parquet_file):
    # Create an empty DataFrame for rankings if the file does not exist
    pd.DataFrame(columns=["roleplay_id", "dataset", "title", "message", *features]).to_parquet(rankings_parquet_file, index=False)

# Helper to save rankings to the Parquet file
def save_rankings_to_parquet(rankings):
    df = pd.DataFrame(rankings)
    print(df.head())
    print(len(df))
    df.to_parquet(rankings_parquet_file, index=False)
# Helper to save rankings to the Parquet file
def save_rankings_to_csv(rankings):
    df = pd.DataFrame(rankings)
    print(df.head())
    print(len(df))
    df.to_csv(rankings_csv_file, index=False)
# Load existing rankings from the Parquet file
def load_rankings_from_csv():
    return pd.read_csv(rankings_csv_file)

def load_rankings_from_parquet():
    return pd.read_parquet(rankings_parquet_file)

rankings_df = load_rankings_from_parquet()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        roleplay_id = int(request.form['roleplay_id'])
        roleplay = next((rp for rp in roleplays if rp['id'] == roleplay_id), None)
        if roleplay:
            scores = {feature: float(request.form.get(feature, 0)) for feature in features}
            # Append new ranking to the existing DataFrame
            new_ranking = {"roleplay_id": roleplay_id, "dataset": roleplay['dataset'], "title": roleplay['title'], "message": roleplay['message'] , **scores}
            rankings_df.loc[len(rankings_df)] = new_ranking
            save_rankings_to_csv(rankings_df)
            save_rankings_to_parquet(rankings_df)

    # Iterate over the rankings to get the next roleplay
    next_roleplay_id = len(rankings_df) % len(roleplays)
    current_roleplay = roleplays[next_roleplay_id]
    print(json.dumps(current_roleplay['message'], indent=0))
    # where role is user message is a list where it is a dict with content and role
    user_message =  [message['content'] for message in current_roleplay['message'] if message['role'] == 'user'][0]
    assistant_message = [message['content'] for message in current_roleplay['message'] if message['role'] == 'assistant'][0]
    # get what number we are at out of the total number of roleplays
    roleplay_number = len(rankings_df)
    return render_template('index.html', roleplay=current_roleplay, assistant_message=assistant_message, user_message=user_message, features=features, roleplay_id=current_roleplay['id'], roleplay_number=roleplay_number)

@app.route('/rankings', methods=['GET'])
def view_rankings():
    # Return the rankings as JSON
    return rankings_df.to_json(orient="records")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8085)
