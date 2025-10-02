from fastapi import FastAPI
from pydantic import BaseModel
from backend.models.emotion_analyzer import analyze_emotion
from backend.models.model_updation import add_feedback
from backend.models.text_genration import chat_with_bot
from typing import List, Optional
from fastapi.middleware.core import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    generated = chat_with_bot(emotion, input.text)
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
