import re
import spacy
import sqlite3
import json
from datetime import datetime
from typing import Dict, Optional
from word2number import w2n

nlp = spacy.load("en_core_web_sm")

conn = sqlite3.connect('./backend/data/user_memory.db', check_same_thread=False)
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS user_profiles (
        session_id TEXT PRIMARY KEY,
        name TEXT,
        age INTEGER,
        location TEXT,
        created_at TEXT,
        updated_at TEXT
    )
''')
c.execute('''
    CREATE TABLE IF NOT EXISTS user_facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        fact TEXT,
        timestamp TEXT
    )
''')
conn.commit()


AGE_KEYWORDS_RE = re.compile(r"\b(i'm|i am|im|my age is|age|i turned)\b", flags=re.I)
DIGIT_RE = re.compile(r"\d+")

def get_profile(session_id: str) -> Dict:
    c.execute("SELECT name, age, location FROM user_profiles WHERE session_id = ?", (session_id,))
    row = c.fetchone()
    if row:
        return {"name": row[0] or "there", "age": row[1], "location": row[2]}
    return {"name": "there", "age": None, "location": None}

def extract_and_save_user_info(session_id: str, text: str):
    doc = nlp(text.lower())
    data = get_profile(session_id)

    # Extract name
    if not data["name"]:
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                data["name"] = ent.text.title()
                updated = True
                break
        if "my name is" in text.lower():
            parts = text.lower().split("my name is")
            if len(parts) > 1:
                possible_name = parts[-1].strip().split()[0]
                data["name"] = possible_name.title()
                updated = True
        if "call me" in text.lower():
            parts = text.lower().split("call me")
            if len(parts) > 1:
                possible_name = parts[-1].strip().split()[0]
                data["name"] = possible_name.title()
                updated = True

    # Extract age
    if not data["age"]:
        if AGE_KEYWORDS_RE.search(text):
            # Digit based age extraction
            for m in DIGIT_RE.finditer(text.lower()):
                age = int(m.group())
                if 10 <= age <= 100:
                    data["age"] = age
                    updated = True
                    break
            # Word based age extraction    
            if not data["age"]:    
                try:
                    may_be_age = w2n.word_to_num(text)
                    if 10 <= may_be_age <= 100:
                        data["age"] = may_be_age
                        updated = True
                except:
                    pass

    # Extract location
    if not data["location"]:
        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC"):
                data["location"] = ent.text.title()
                updated = True
                break

    # Save other info
    if not data["name"] or not data["age"] or not data["location"]:
        c.execute("INSERT INTO user_facts (session_id, fact, timestamp) VALUES (?, ?, ?)",
                   (session_id, text[:200], datetime.now().isoformat()))
        conn.commit()

    # Update profile if new info found
    if updated:
        c.execute("""
            INSERT OR REPLACE INTO user_profiles 
            (session_id, 
                name, 
                age, 
                location, 
                created_at, 
                updated_at)
            VALUES (?, ?, ?, ?, COALESCE((SELECT created_at FROM user_profiles WHERE session_id=?), ?), ?)
        """, (
            session_id,
            data["name"],
            data["age"],
            data["location"],
            datetime.now().isoformat(),
            datetime.now().isoformat()
        ))
        conn.commit()



        





# # model
# model_name = "microsoft/DialoGPT-medium"
# device = torch.device("cpu")
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float32,
#     low_cpu_mem_usage=True).to(device)

# # for natural and engaging conversations
# filters = ['hmm', 'you know', 'I see', 'oh', 'well', 'like', 'so']
# def add_filters(text, prob=0.2):
#     if random.random() < prob:
#         filter = random.choice(filters)
#         text = f"{filter}, {text}"
#         return text
#     return text

# def extract_name(text):
#     doc = nlp(text)
#     for ent in doc.ents:
#         if ent.label_ == "PERSON":
#             return ent.text
#     return None

# def chat_with_bot(emotion, user_input, user_id='User'):
#     # if history is None:
#     #     history = []

#     emotion_label = [e['label'] for e in emotion]
#     name = extract_name(user_input) or user_id


#     prompt = "You are a friendly and empathetic chatbot named Goru. Speak naturally and emotionally based on the user's feelings."
#     top_emotions = emotion_label[:2]
#     prompt += f" Given emotions: {', '.join(top_emotions)}."

#     # emotional tone adjustment
#     emotional_tone_map = {
#         'joy': "cheerful",
#         'amusement': "cheerful",
#         'excitement': "cheerful",
#         'sadness': "empathetic",
#         'grief': "empathetic",
#         'disappointment': "empathetic",
#         'remorse': "empathetic",
#         'anger': "calm",
#         'annoyance': "calm",
#         'disapproval': "calm",
#         'disgust': "calm",
#         'fear': "reassuring",
#         'nervousness': "reassuring",
#         'love': "warm",
#         'caring': "warm",
#         'admiration': "warm",
#         'confusion': "helpful",
#         'realization': "helpful",
#         'curiosity': "friendly",
#         'desire': "friendly",
#         'approval': "positive",
#         'gratitude': "positive",
#         'relief': "positive",
#         'optimism': "positive",
#         'pride': "positive",
#         'embarrassment': "gentle",
#         'surprise': "curious",
#         'none': "neutral"
#     }


#     for e in top_emotions:
#         if e in emotional_tone_map:
#             prompt += " " + emotional_tone_map[e]

#     # check for conversation history
#     # if history:
#     #     prompt += " Previous conversation:\n"
#     #     for msg in history[-5:]:
#     #         prompt += f"User: {msg['role']}\nAI: {msg['content']}\n"

#     # Add current user input
#     prompt += f"\n{name}: {user_input}\nAI:"

#     inputs = tokenizer(
#         prompt,
#         return_tensors="pt",
#         truncation=True
#     ).to(device)

#     with torch.no_grad():
#         output = model.generate(
#             **inputs,
#             max_length=512,
#             temperature=0.7,
#             top_k=50,
#             top_p=0.9,
#             do_sample=True,    # Creative, varied text generation
#             num_return_sequences=1,
#             repetition_penalty=1.2,
#             pad_token_id=tokenizer.eos_token_id,
#             num_beams=5
#         )

#     response = tokenizer.decode(output[0], skip_special_tokens=True)
#     response = response.split("Goru:")[-1].strip()
#     response = add_filters(response, prob=0.2)

#     return response
    
# examples
# print(chat_with_bot([{'label': 'joy'}], "Hey Goru! I got selected for my internship!"))
# print(chat_with_bot([{'label': 'sadness'}], "I miss someone who doesn’t talk to me anymore."))
# print(chat_with_bot([{'label': 'curiosity'}], "Why do people overthink so much, Goru?"))
# print(chat_with_bot([{'label': 'anger'}], "I'm so annoyed, my project group never listens to my ideas!"))
# print(chat_with_bot([{'label': 'love'}], "I really admire someone, Goru. They make everything feel peaceful."))
# print(chat_with_bot([{'label': 'fear'}], "I have my presentation tomorrow, I’m so anxious."))
# print(chat_with_bot([{'label': 'confusion'}], "I'm not sure if I chose the right path in life."))
# print(chat_with_bot([{'label': 'gratitude'}], "Thanks for listening to me, Goru."))
# print(chat_with_bot([{'label': 'optimism'}], "I feel like things are finally starting to make sense now."))
# print(chat_with_bot([{'label': 'embarrassment'}], "I tripped in front of everyone today."))
# print(chat_with_bot([{'label': 'none'}], "What’s your favorite thing about conversations, Goru?"))
# print(chat_with_bot([{'label': 'love'}], "Sometimes I feel emotions too deeply. It’s beautiful but exhausting."))


# =======================================================================

# import os
# import openai
# openai.api_key = "OPENAI_API_KEY"    # set your OpenAI API key in terminal
# # openai.api_key = os.getenv("OPENAI_API_KEY")
# start_sequence = "\nAI:"
# restart_sequence = "\nHuman: "
# response = openai.Completion.create(
#   model="gpt-3.5-turbo-instruct",
#   prompt="The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: ",
#   temperature=0.9,
#   max_tokens=150,
#   top_p=1,
#   frequency_penalty=0,
#   presence_penalty=0.6,
#   stop=[" Human:", " AI:"]
# )