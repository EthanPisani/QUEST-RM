# auto_rank_batching.py
import requests
import json
import pandas as pd
import os
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
import time

# model = "deepseek-ai_DeepSeek-R1-Distill-Llama-8B-exl2"
# model = "QwQ-32B-exl2"
model = "DeepSeek-R1-Distill-Qwen-14B-exl2"

OPENAI_API_KEY = ""
OPENAI_API_URL = "http://127.0.0.1:5009/v1/completions"

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
# rankings_parquet_file = "auto_rankings1.parquet"
rankings_parquet_file = "auto_rankings5.parquet"

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
        print(rankings)

        resume_index = len(rankings)
        return rankings
    return pd.DataFrame(columns=["roleplay_id", "dataset", "title", "message", *features])

def save_rankings(rankings_df):
    rankings_df.to_parquet(rankings_parquet_file, index=False)

rankings_df = load_rankings()

def load_human_rankings():
    if os.path.exists(human_rankings_parquet_file):
        hdf = pd.read_parquet(human_rankings_parquet_file)
        hdf = hdf[~(hdf.select_dtypes(include='number') == 0).any(axis=1)]
        return hdf
    return pd.DataFrame(columns=["roleplay_id", *features])

human_rankings_df = load_human_rankings()
example_prompts = []

def update_examples():
    global example_prompts
    if len(human_rankings_df) >= 15:
        example_prompts = human_rankings_df.sample(n=8).to_dict(orient='records')

update_examples()

def rank_roleplay(roleplay):
    global example_prompts
    user_message = next((msg['content'] for msg in roleplay['message'] if msg['role'] == 'user'), "")
    assistant_message = next((msg['content'] for msg in roleplay['message'] if msg['role'] == 'assistant'), "")
    roleplay_id = roleplay['id']
    # print(f"Processing roleplay {roleplay_id}")
    # print(example_prompts)
    # print(len(str(example_prompts)))
    prompt = f"""
You are an expert in roleplay analysis. Given the following roleplay interaction, evaluate the assistant's response based on six criteria:
. Contextual Alignment

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

3. Descriptive Length and Depth

Definition:
This evaluates the richness and length of descriptions in the roleplay text. Ideally, descriptions should be about 4 to 6 sentences—long enough to be immersive and detailed but not overly drawn out. A well-balanced description enhances engagement and allows for a deep, conversational style.

What to Look For:

    Balanced Length: Is the description within the 4 to 6 sentence range, providing enough detail without becoming excessive or too brief?
    Sensory Details: Does it incorporate sight, sound, touch, smell, or taste to create a vivid scene?
    Environmental Awareness: Does the response acknowledge and interact with the surroundings rather than feeling empty?

How to Rank:

    High: Detailed and immersive descriptions within the ideal length, making scenes vivid and engaging.
    Mid: Some detail is present, but the description may be slightly too short or too long for smooth readability.
    Low: Minimal or overly brief descriptions that fail to create a strong sense of place or emotion.

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

Tips for Evaluating Roleplay Descriptions:

    Watch for OOC (Out of Character) Text: If a message includes non-roleplay elements like discussions about game mechanics, player chatter, or real-world topics, it should not be rated for descriptive depth.
    Avoid Casual or Generic Replies: Simple acknowledgments (e.g., "Okay," "Sounds good") or short, vague responses without descriptive elements are not true roleplay descriptions.
    Look for Intentional Roleplay Writing: A proper roleplay message should contain some level of narrative, character action, or environmental detail rather than just being a standard conversation.

Here are a list of past examples of ranked interactions:
{example_prompts}

Title: {roleplay['title']}
User message: {user_message}
Assistant response: {assistant_message}

Please return a JSON object with scores (from 1.0 to 10.0) for:
{features}
Sure! Here is a JSON object with scores for the six criteria: """
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 40,
        "n": 1,
        "stop": None,
        "stream": False,
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
        scores_json = json.loads(result["choices"][0]["text"])
        return roleplay, scores_json
    except requests.exceptions.RequestException as err:
        print(f"Ranking request failed: {err}")
        print(response.text)
        return roleplay, None
    except json.JSONDecodeError as err:
        print(f"Failed to decode JSON response: {err}")
        print(response.text)
        return roleplay, None

def main():
    global resume_index
    request_count = 0
    bar = tqdm.tqdm(total=len(roleplays))
    if resume_index > 0:
        bar.update(resume_index)
        bar.set_description(f"Resuming from index {resume_index}")
        request_count = resume_index

    batch_size = 16  # Adjust batch size as needed
    range_size = 128
    last_saved_range = request_count // range_size
    failed_requests = {}
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        start_time = time.perf_counter()

        for roleplay in roleplays[resume_index:]:

            futures.append(executor.submit(rank_roleplay, roleplay))
            if len(futures) >= batch_size:
                for future in as_completed(futures):
                    roleplay_scored, scores = future.result()
                    if scores:
                        new_ranking = {"roleplay_id": roleplay_scored['id'], "dataset": roleplay_scored['dataset'], "title": roleplay_scored['title'], "message": roleplay_scored['message'], **scores}
                        print(scores)
                        rankings_df.loc[len(rankings_df)] = new_ranking
                        request_count += 1
                    if scores is None: # resubmit the request
                        print("Resubmitting request")
                        futures.append(executor.submit(rank_roleplay, roleplay_scored))
                        # add 1 to dict of failed requests on id
                        failed_requests[roleplay_scored['id']] = failed_requests.get(roleplay_scored['id'], 0) + 1
                        # if any above 3, terminate
                        if failed_requests[roleplay_scored['id']] > 3:
                            print(f"Failed to process roleplay {roleplay_scored['id']} after 3 attempts. Ending program.")
                            save_rankings(rankings_df)
                            return
                    bar.update(1)
                    bar.set_description(f"Processing roleplay {roleplay_scored['id']}")
                futures = []
                end = time.perf_counter()
                print(f"Processed {request_count-resume_index} roleplays in {end - start_time:0.4f} seconds.")
                current_range = request_count // range_size
                if current_range != last_saved_range:
                    print(f"Entered new 1000-count range: {current_range * range_size}-{(current_range + 1) * range_size - 1}")
                    save_rankings(rankings_df)
                    update_examples()
                    last_saved_range = current_range 

        # Process any remaining futures
        for future in as_completed(futures):
            roleplay_scored, scores = future.result()
            if scores:
                new_ranking = {"roleplay_id": roleplay_scored['id'], "dataset": roleplay_scored['dataset'], "title": roleplay_scored['title'], "message": roleplay_scored['message'], **scores}
                rankings_df.loc[len(rankings_df)] = new_ranking
                request_count += 1
            bar.update(1)
            bar.set_description(f"Processing roleplay {roleplay_scored['id']}")

    save_rankings(rankings_df)
    print(f"Ranking process complete in {end - start_time:0.4f} seconds.")
    bar.close()

if __name__ == "__main__":
    main()
