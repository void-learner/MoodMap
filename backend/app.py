from fastapi import FastAPI
from pydantic import BaseModel
from models.emotion_analyzer import analyze_emotion
from models.model_updation import add_feedback
from models.text_genration import chat_with_bot
from typing import List, Optional

app = FastAPI()

class Input(BaseModel):
    text: str

class Feedback(BaseModel):
    text: str
    true_labels: List[str]
    predicted_labels: Optional[List[str]] = None
    is_correct: bool

@app.post("/analyze_emotion")
def analyze(input: Input):
    emotion = analyze_emotion(input.text)
    generated = chat_with_bot(input.text, emotion)
    return {'emotion': emotion, 'generated_text': generated}

@app.post("/feedback")
def feedback(feedback: Feedback):
    add_feedback(
        feedback.text,
        feedback.true_labels,
        feedback.predicted_labels,
        feedback.is_correct
    )
    return {'status': 'Feedback received'}
