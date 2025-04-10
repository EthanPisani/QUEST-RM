import requests
import json
import pandas as pd
import os
from pydantic import BaseModel
model = "DeepSeek-R1-Distill-Qwen-14B-exl2"
OPENAI_API_KEY = "f2470b56e3d23f2e52327eac74445f36"
OPENAI_API_URL = "http://10.0.9.12:5008/v1/completions"

features = [
    "Contextual_Alignment",
    "Character_Consistency",
    "Descriptive_Depth",
    "Role_Specific_Knowledge",
    "Engagement_and_Collaboration",
    "Creativity_and_Emotional_Nuance"
]
class RankingRequest(BaseModel):
    Contextual_Alignment: float
    Character_Consistency: float
    Descriptive_Depth: float
    Role_Specific_Knowledge: float
    Engagement_and_Collaboration: float
    Creativity_and_Emotional_Nuance: float
parquet_file = "processed/sample_dataset_2.parquet"
rankings_parquet_file = "auto_rankings1.parquet"
human_rankings_parquet_file = "rankings.parquet"

def load_roleplays():
    df = pd.read_parquet(parquet_file)
    return [
        {
            "id": row.name,
            "dataset": row['dataset'],
            "title": row['title'],
            "message": json.loads(row['message'])
        }
        for _, row in df.iterrows()
    ]


roleplays = load_roleplays()


section = 0
roleplays = roleplays[section * 100000:(section + 1) * 100000]
resume_index = 0

def load_rankings():
    global resume_index
    if os.path.exists(rankings_parquet_file):
        rankings = pd.read_parquet(rankings_parquet_file)
        resume_index = len(rankings)
        return rankings
    return pd.DataFrame(columns=["roleplay_id", "dataset", "title", "message", *features])


def save_rankings(rankings_df):
    rankings_df.to_parquet(rankings_parquet_file, index=False)


rankings_df = load_rankings()


def load_human_rankings():
    if os.path.exists(human_rankings_parquet_file):
        return pd.read_parquet(human_rankings_parquet_file)
    return pd.DataFrame(columns=["roleplay_id", *features])


human_rankings_df = load_human_rankings()
example_prompts = []

def update_examples():
    global example_prompts
    if len(human_rankings_df) >= 15:
        example_prompts = human_rankings_df.sample(n=15).to_dict(orient='records')

update_examples()

def rank_roleplay(roleplay, request_count):
    user_message = next((msg['content'] for msg in roleplay['message'] if msg['role'] == 'user'), "")
    assistant_message = next((msg['content'] for msg in roleplay['message'] if msg['role'] == 'assistant'), "")
    
    def request_ranking():
        prompt = f"""
You are an expert in roleplay analysis. Given the following roleplay interaction, evaluate the assistant's response based on six criteria:
1. Contextual Alignment

Definition:
Contextual alignment measures how well the roleplay text fits within the ongoing scene, prior messages, and the established lore or worldbuilding.

What to Look For:

    Continuity: Does the response logically follow from the previous messages? Are there inconsistencies or contradictions?
    Worldbuilding Adherence: If the RP occurs in a specific universe (e.g., medieval fantasy, sci-fi, cyberpunk), does the text stay true to the setting's rules and logic?
    Situational Awareness: Does the character appropriately react to environmental cues, ongoing conflicts, or major story events?

How to Rank:

    High: The response seamlessly builds on prior exchanges, respects world rules, and demonstrates clear situational awareness.
    Mid: Some minor inconsistencies or slight misinterpretations of the setting, but generally coherent.
    Low: The response ignores prior messages, breaks established rules, or derails the scene.

2. Character Consistency

Definition:
This assesses whether the character stays true to their personality, goals, and previously established traits.

What to Look For:

    Dialogue Authenticity: Does the character’s speech pattern, vocabulary, and tone match how they’ve spoken previously?
    Behavioral Coherence: Are the character’s actions consistent with their backstory, motivations, and past decisions?
    Emotional Consistency: If the character was previously fearful, confident, or hesitant, does their emotional state progress naturally rather than flipping arbitrarily?

How to Rank:

    High: Character behaves consistently across dialogue and actions, with believable development.
    Mid: Minor inconsistencies that don’t drastically break immersion (e.g., slight deviations in speech style).
    Low: The character acts out of character (e.g., a stoic warrior suddenly becomes playful with no explanation).

3. Descriptive Depth

Definition:
This evaluates the richness of descriptions in the roleplay text, which enhances immersion and visualization.

What to Look For:

    Sensory Details: Are sight, sound, touch, smell, or taste used to paint a vivid picture?
    Environmental Interaction: Does the response acknowledge surroundings, rather than existing in a void?
    Body Language & Microexpressions: Do characters’ movements and subtle expressions add depth to their emotions?

How to Rank:

    High: Engaging, multi-sensory descriptions that make scenes and actions vivid.
    Mid: Some descriptive elements, but could be expanded for better immersion.
    Low: Minimal or no descriptive detail, making the text feel bland or detached.

4. Role-Specific Knowledge

Definition:
This measures how well the response reflects expertise or understanding of the character’s profession, skills, or setting.

What to Look For:

    Specialized Knowledge: If playing a scientist, are their explanations scientifically sound (or at least believable within the setting)?
    Combat Realism: If playing a warrior, do their actions reflect proper tactics or weapon knowledge?
    Social & Cultural Nuance: Does the character’s background inform their choices (e.g., a noble speaking differently than a street thief)?

How to Rank:

    High: The character’s knowledge and actions feel authentic, enhancing believability.
    Mid: Some level of expertise is present but lacks depth or realism.
    Low: The character displays ignorance of their supposed expertise, breaking immersion.

5. Engagement and Collaboration

Definition:
This examines how well the response fosters dynamic interactions, giving other players opportunities to contribute.

What to Look For:

    Open-Ended Prompts: Does the text invite others to respond (e.g., asking questions, leaving room for reactions)?
    Scene Advancement: Does the response move the plot forward rather than stalling?
    Avoidance of Powerplay: Does the character avoid dictating other players’ actions without consent?

How to Rank:

    High: The response enhances collaboration and keeps the story engaging.
    Mid: The character engages, but their actions may be too passive or slightly limiting.
    Low: The response is self-contained, ignores others, or dominates the scene without room for input.

6. Creativity and Emotional Nuance

Definition:
This assesses originality, depth of emotional expression, and overall impact of the response.

What to Look For:

    Inventiveness: Does the response introduce unique ideas, twists, or approaches?
    Emotional Subtlety: Are emotions conveyed naturally through dialogue, thoughts, and actions?
    Impactfulness: Does the response leave a memorable impression, evoking strong imagery or emotions?

How to Rank:

    High: The response is imaginative, emotionally rich, and leaves a strong impact.
    Mid: Some creativity and emotion, but could be more engaging.
    Low: Bland, predictable, or emotionally flat response.
Here are a list of past examples of ranked interactions:
{example_prompts}

User message: {user_message}
Assistant response: {assistant_message}

Please return a JSON object with scores (from 1.0 to 10.0) for:
{features}
        """
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
            # print("Raw ranking result:", result)
            scores_json = json.loads(result["choices"][0]["text"])
            return scores_json
        except requests.exceptions.RequestException as err:
            print(f"Ranking request failed: {err}")
            print(response.text)

            return None
        except json.JSONDecodeError as err:
            print(f"Failed to decode JSON response: {err}")
            print(response.text)
            return None
    
    scores = None
    while not scores or all(v == 5 for v in scores.values()) or all(v == 10 for v in scores.values()):
        scores = request_ranking()
    
    print(f"Final Ranking for roleplay {roleplay['id']}: {scores}")
    
    if request_count % 1000 == 0:
        update_examples()

    
    return scores
import tqdm 
def main():
    global resume_index
    request_count = 0
    bar = tqdm.tqdm(total=len(roleplays))
    # load current rankings
    if resume_index > 0:
        bar.update(resume_index)
        bar.set_description(f"Resuming from index {resume_index}")
        request_count = resume_index
        

    for roleplay in roleplays[resume_index:]:
        bar.update(1)
        bar.set_description(f"Processing roleplay {roleplay['id']}")
        scores = rank_roleplay(roleplay, request_count)
        if scores:
            new_ranking = {"roleplay_id": roleplay['id'], "dataset": roleplay['dataset'], "title": roleplay['title'], "message": roleplay['message'], **scores}
            rankings_df.loc[len(rankings_df)] = new_ranking
            request_count += 1
        if request_count % 1000 == 0:
            save_rankings(rankings_df)
    
    save_rankings(rankings_df)
    print("Ranking process complete.")

if __name__ == "__main__":
    main()
