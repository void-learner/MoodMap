import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))


from fastapi import FastAPI
from pydantic import BaseModel
from backend.models.emotion_analyzer import analyze_emotion, train_emotion_model
from backend.models.model_updation import add_feedback
from backend.models.text_genration import chat_with_bot
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import ollama
import uuid

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


class Input(BaseModel):
    text: str
    session_id: Optional[str] = None

class Feedback(BaseModel):
    text: str
    true_labels: List[str]
    predicted_labels: Optional[List[str]] = None
    is_correct: bool

@app.post("/analyze_emotion")
def analyze(input: Input):
    session_id = input.session_id or str(uuid.uuid4())

    emotions = analyze_emotion(input.text)
    reply = chat_with_bot(input.text, emotions, session_id)

    return {
        "emotions": emotions,
        "generated_text": reply,
        "session_id": session_id
    }

@app.post("/feedback")
def feedback(feedback: Feedback):
    add_feedback(
        feedback.text,
        feedback.true_labels,
        feedback.predicted_labels,
        feedback.is_correct
    )
    return {'status': 'Feedback received'}

# For on-demand training
# @app.post("/train_model")
# def train():
#     result = train_emotion_model()
#     return {'status': result}
