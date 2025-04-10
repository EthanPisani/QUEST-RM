# rank_roleplay_llm_async.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
import os
import requests
import threading
from pydantic import BaseModel

app = Flask(__name__)
# model = "DeepSeek-R1-Distill-Qwen-14B-exl2"
model = "deepseek-ai_DeepSeek-R1-Distill-Llama-8B-exl2"
# API URL
OPENAI_API_KEY = ""
OPENAI_API_URL = "http://10.0.9.12:5008/v1/completions"
# List of features to rank
features = [
    "Contextual Alignment",
    "Character Consistency",
    "Descriptive Depth",
    "Role-Specific Knowledge",
    "Engagement and Collaboration",
    "Creativity and Emotional Nuance"
]
# List of ranking features
features = [
    "Contextual Alignment",
    "Character Consistency",
    "Descriptive Depth",
    "Role-Specific Knowledge",
    "Engagement and Collaboration",
    "Creativity and Emotional Nuance"
]

# Path to roleplay dataset
parquet_file = "processed/sample_dataset.parquet"

def load_roleplays():
    df = pd.read_parquet(parquet_file)
    roleplays = [
        {
            "id": row.name,
            "dataset": row['dataset'],
            "title": row['title'],
            "message": json.loads(row['message'])
        }
        for _, row in df.iterrows()
    ]
    return roleplays

roleplays = load_roleplays()

# Rankings storage
rankings_parquet_file = "rankings.parquet"

if not os.path.exists(rankings_parquet_file):
    pd.DataFrame(columns=["roleplay_id", "dataset", "title", "message", *features]).to_parquet(rankings_parquet_file, index=False)

def save_rankings(rankings_df):
    rankings_df.to_parquet(rankings_parquet_file, index=False)

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

def rank_roleplay(roleplay_id):
    """Runs ranking asynchronously and updates the database"""
    roleplay = next((rp for rp in roleplays if rp['id'] == roleplay_id), None)
    if not roleplay:
        return

    user_message = next((msg['content'] for msg in roleplay['message'] if msg['role'] == 'user'), "")
    assistant_message = next((msg['content'] for msg in roleplay['message'] if msg['role'] == 'assistant'), "")
    
    examples = rankings_df.sample(n=8).to_dict(orient='records') if len(rankings_df) >= 8 else rankings_df.to_dict(orient='records')

    prompt = f"""
    You are an expert in roleplay analysis. Given the following roleplay interaction, evaluate the assistant's response based on six criteria:
    
    Here are five past examples of ranked interactions:
    {expand_ranked(examples)}

    User message: {user_message}
    Assistant response: {assistant_message}



    Please return a JSON object with scores (from 1.0 to 10.0) for the following categories:
    {features}
    """


    # Prepare the ranking request
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 200,
        "temperature": 0.9,
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
        response.raise_for_status()
        result = response.json()
        print("Raw ranking result:", result)    
        scores_json = json.loads(result["choices"][0]["text"])
        print(f"Ranking scores for roleplay {roleplay_id}:", scores_json )
        return scores_json

    except requests.exceptions.RequestException as err:
        print(f"Ranking request failed: {err}")

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
    """Serve roleplay text first with default values"""
    next_roleplay_id = len(rankings_df) % len(roleplays)
    current_roleplay = roleplays[next_roleplay_id]
    user_message = [msg['content'] for msg in current_roleplay['message'] if msg['role'] == 'user'][0]
    assistant_message = [msg['content'] for msg in current_roleplay['message'] if msg['role'] == 'assistant'][0]

    features_feilds = RankingRequest.model_fields.keys()
    roleplay_number = len(rankings_df)
    return render_template('async_ranking.html',
                            roleplay=current_roleplay,
                            assistant_message=assistant_message,
                            user_message=user_message,
                            features=features_feilds,
                            roleplay_id=current_roleplay['id'],
                            roleplay_number=roleplay_number)

@app.route('/rank/<int:roleplay_id>', methods=['GET'])
def rank(roleplay_id):
    """Trigger ranking in background"""
    scores = rank_roleplay(roleplay_id)
    print("Ranking scores:", scores)
    return jsonify(scores)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9301)
