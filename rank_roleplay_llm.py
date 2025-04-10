from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
import os
import requests
import random
from pydantic import BaseModel

app = Flask(__name__)

# OpenAI API Key (Ensure this is properly secured in production)
OPENAI_API_KEY = "f2470b56e3d23f2e52327eac74445f36"
OPENAI_API_URL = "http://127.0.0.1:5008/v1/completions"

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

def load_roleplays():
    df = pd.read_parquet(parquet_file)
    roleplays = []
    for _, row in df.iterrows():
        roleplays.append({
            "id": row.name,
            "dataset": row['dataset'],
            "title": row['title'],
            "message": json.loads(row['message'])
        })
    return roleplays

roleplays = load_roleplays()

# Paths for saving rankings
rankings_parquet_file = "rankings.parquet"
rankings_csv_file = "rankings.csv"

if not os.path.exists(rankings_csv_file):
    pd.DataFrame(columns=["roleplay_id", "dataset", "title", "message", *features]).to_csv(rankings_csv_file, index=False)
    
if not os.path.exists(rankings_parquet_file):
    pd.DataFrame(columns=["roleplay_id", "dataset", "title", "message", *features]).to_parquet(rankings_parquet_file, index=False)

def save_rankings(rankings):
    df = pd.DataFrame(rankings)
    df.to_csv(rankings_csv_file, index=False)
    df.to_parquet(rankings_parquet_file, index=False)

def load_rankings():
    return pd.read_parquet(rankings_parquet_file)

rankings_df = load_rankings()

class RankingRequest(BaseModel):
    Contextual_Alignment: float
    Character_Consistency: float
    Descriptive_Depth: float
    Role_Specific_Knowledge: float
    Engagement_and_Collaboration: float
    Creativity_and_Emotional_Nuance: float

def expand_ranked(examples):
    expanded = "\n".join([f"{ex['roleplay_id']}: {ex['title']} - {ex['dataset']}\n{ex['message']}" for ex in examples])
    print(expanded)
    return expanded

def rank_response(user_message, assistant_message, examples):
    """Uses an AI API to rank the assistant's response based on predefined features."""
    
    prompt = f"""
    You are an expert in roleplay analysis. Given the following roleplay interaction, evaluate the assistant's response based on six criteria:
    
    Here are five past examples of ranked interactions:
    {expand_ranked(examples)}

    User message: {user_message}
    Assistant response: {assistant_message}



    Please return a JSON object with scores (from 1.0 to 10.0) for the following categories:
    {features}
    """

    payload = {
        "model": "Meta-Llama-3.1-8B-Instruct-exl2",
        "prompt": prompt,
        "max_tokens": 200,  # Adjust as needed
        "temperature": 0.3,
        "top_p": 0.9,
        "n": 1,
        "stop": None,
        "json_schema": RankingRequest.model_json_schema()
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise error for bad status codes (e.g., 422)
        
        result = response.json()
        print(result)
        scores_json = json.loads(result["choices"][0]["text"])  # Extract JSON response

        return scores_json

    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
    except requests.exceptions.RequestException as err:
        print(f"Request Error: {err}")
    
    return None


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        roleplay_id = int(request.form['roleplay_id'])
        roleplay = next((rp for rp in roleplays if rp['id'] == roleplay_id), None)
        if roleplay:
            scores = {feature: float(request.form.get(feature, 0)) for feature in features}
            new_ranking = {"roleplay_id": roleplay_id, "dataset": roleplay['dataset'], "title": roleplay['title'], "message": roleplay['message'], **scores}
            rankings_df.loc[len(rankings_df)] = new_ranking
            save_rankings(rankings_df)
    
    next_roleplay_id = len(rankings_df) % len(roleplays)
    current_roleplay = roleplays[next_roleplay_id]
    user_message = [msg['content'] for msg in current_roleplay['message'] if msg['role'] == 'user'][0]
    assistant_message = [msg['content'] for msg in current_roleplay['message'] if msg['role'] == 'assistant'][0]
    past_examples = rankings_df.sample(n=2).to_dict(orient='records') if len(rankings_df) >= 2 else rankings_df.to_dict(orient='records')

    features_and_scores = rank_response(user_message, assistant_message, past_examples)
    roleplay_number = len(rankings_df)
    # replace _ with space for suggestions
    # features_and_scores = {feature.replace("_", " "): suggested_scores[feature] for feature in features}
    print("Suggested scores:", features_and_scores)
    
    return render_template('index.html', 
                            roleplay=current_roleplay, 
                            assistant_message=assistant_message, 
                            user_message=user_message, 
                            features_and_scores=features_and_scores,
                            roleplay_id=current_roleplay['id'], 
                            roleplay_number=roleplay_number)

@app.route('/rankings', methods=['GET'])
def view_rankings():
    return rankings_df.to_json(orient="records")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8085)
