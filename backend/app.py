from fastapi import FastAPI
from pydantic import BaseModel
from backend.models.emotion_analyzer import analyze_emotion, train_emotion_model
from backend.models.model_updation import add_feedback
from backend.models.text_genration import extract_and_save_user_info, get_profile
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import ollama


app = FastAPI()

# origins = [
#     "http://localhost",
#     "http://127.0.0.1",
#     "http://localhost:5173",
#     "http://127.0.0.1:5173",
#     "file://", 
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

history = {}

class Input(BaseModel):
    text: str
    session_id: str

class Feedback(BaseModel):
    text: str
    true_labels: List[str]
    predicted_labels: Optional[List[str]] = None
    is_correct: bool

@app.post("/analyze_emotion")
def analyze(input: Input):
    session_id = input.session_id

    profile = extract_and_save_user_info(session_id, input.text)
    name = profile["name"]

    system_prompt = f"""You are a warm, caring friend talking to {name}.
    {f'{name} is {profile["age"]} years old.' if profile["age"] else ''}
    {f'They live in {profile["location"]}.' if profile["location"] else ''}
    Be casual, supportive, and use their name naturally. Remember what they’ve told you before."""

    messages = [{"role": "system", "content": system_prompt}]
    messages += history[-10:]  
    messages.append({"role": "user", "content": input.text})

    response = ollama.chat(model="llama2:7b", messages=messages)
    bot_reply = response["message"]["content"]

    history.append({"role": "user", "content": input.text})
    history.append({"role": "assistant", "content": bot_reply})
    history[session_id] = history

    emotion = analyze_emotion(input.text, version=None)
    return {"session_id": session_id, 'emotion': emotion, 'generated_text': bot_reply}

@app.post("/feedback")
def feedback(feedback: Feedback):
    add_feedback(
        feedback.text,
        feedback.true_labels,
        feedback.predicted_labels,
        feedback.is_correct
    )
    return {'status': 'Feedback received'}

@app.post("/clear_chat")
def clear_chat(session_id: str):
    history.pop(session_id, None)
    return {"status": "cleared"}

# For on-demand training
# @app.post("/train_model")
# def train():
#     result = train_emotion_model()
#     return {'status': result}
