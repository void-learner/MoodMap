import spacy
import sqlite3
import re 
from word2number import w2n
from datetime import datetime
from typing import Dict

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

conn = sqlite3.connect("./backend/data/user_memory.db", check_same_thread=False)
c = conn.cursor()

# Create tables 
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

AGE_KEYWORDS_RE = re.compile(r"\b(i'?m|i am|im|my age is|age|i turned)\b", re.I)
DIGIT_RE = re.compile(r"\d+")

def get_profile(session_id: str) -> Dict:
    c.execute("SELECT name, age, location FROM user_profiles WHERE session_id = ?", (session_id,))
    row = c.fetchone()
    if row:
        return {
            "name": row[0] or "there",
            "age": row[1],
            "location": row[2]
        }
    return {"name": "there", "age": None, "location": None}

def extract_and_save_user_info(session_id: str, text: str) -> Dict:
    doc = nlp(text.lower())
    profile = get_profile(session_id)
    updated = False

    # Extract name
    if not profile["name"] or profile["name"] == "there":
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                profile["name"] = ent.text.title()
                updated = True
                break
        if "my name is" in text.lower():
            parts = text.lower().split("my name is")[-1].strip()
            possible_name = re.split(r'[\s\.\,\!\?]', parts)[0]
            if len(parts) > 1:
                profile["name"] = possible_name.title()
                updated = True
        if "call me" in text.lower():
            parts = text.lower().split("call me")[-1].strip()
            possible_name = re.split(r'[\s\.\,\!\?]', parts)[0]
            if len(parts) > 1:
                profile["name"] = possible_name.title()
                updated = True

    # Extract age
    if not profile["age"]:
        if AGE_KEYWORDS_RE.search(text):
            # Digit based age extraction
            for m in DIGIT_RE.finditer(text.lower()):
                age = int(m.group())
                if 10 <= age <= 100:
                    profile["age"] = age
                    updated = True
                    break
            # Word based age extraction    
            if not profile["age"]:    
                try:
                    may_be_age = w2n.word_to_num(text)
                    if 10 <= may_be_age <= 100:
                        profile["age"] = may_be_age
                        updated = True
                except:
                    pass

    # Extract location
    if not profile["location"]:
        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC") and len(ent.text) > 2:
                profile["location"] = ent.text.title()
                updated = True
                break 

    # Save interesting fact
    c.execute(
        "INSERT INTO user_facts (session_id, fact, timestamp) VALUES (?, ?, ?)",
        (session_id, text[:300], datetime.now().isoformat())
    )
    conn.commit()

    # Save profile if updated
    if updated:
        c.execute(
            '''
            INSERT OR REPLACE INTO user_profiles
            (session_id, name, age, location, created_at, updated_at)
            VALUES (?, ?, ?, ?, COALESCE((SELECT created_at FROM user_profiles WHERE session_id = ?), ?), ?)
            ''',
            (
                session_id,
                profile["name"],
                profile["age"],
                profile["location"],
                session_id,
                datetime.now().isoformat(),
                datetime.now().isoformat()
            )
        )                   
        conn.commit()

    return profile