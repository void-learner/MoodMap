# MoodMap

A full-stack desktop application for fine-grained emotion analysis using BERT, with user feedback for continuous model improvement, and empathetic text generation.

<img width="480" height="101" alt="image" src="https://github.com/user-attachments/assets/6273486f-af12-4a48-903f-3161e193a00a" />


Description
This project is a BERT-powered chatbot that analyzes user messages for emotions (e.g., joy, sadness, fear) using the GoEmotions dataset. It supports multi-label classification, generates context-aware responses, and incorporates a feedback loop to fine-tune the model over time. The backend is built with FastAPI and Hugging Face Transformers, while the frontend is a desktop app using Vite, React, TypeScript, Electron, Tailwind CSS, and Lucide React icons.
The app features a clean UI with bot introductions, emotion probability displays (e.g., "Neutral 30%"), Yes/No feedback buttons, and smooth chatting experience with auto-scroll and typing indicators.

## Features

Emotion Analysis: Fine-tuned BERT model on GoEmotions for 28+ emotions.
Text Generation: Empathetic responses based on detected emotions using llama.
Feedback Loop: Users provide corrections; model updates via fine-tuning and versioning.
Model Versioning: Saves models in data/saved_models/ with incremental versions.
Desktop UI: Responsive chat interface with Tailwind styling and Lucide icons.
Persistence: SQLite for feedback, Docker for backend containerization.
Development Tools: Hot module replacement (HMR) in dev mode, cross-platform packaging.

## Tech Stack

Backend: Python 3.12, FastAPI, Hugging Face Transformers, PyTorch, scikit-learn, SQLite.
Frontend: Vite, React 18, TypeScript, Electron, Tailwind CSS, Lucide React, Axios.
DevOps: Docker, Docker Compose, Git.
ML: BERT for classification, GPT-like for generation, GoEmotions dataset.

## Prerequisites

Python 3.12+
Node.js 18+ and npm
Docker (for backend containerization)
Git
ollama

## Steps

### step 1: Terminal 1 – Setup ollama (we can use any version)
ollama pull gemma2:2b 

### step 2: Terminal 2 – Start Backend
cd backend
uvicorn app:app --reload --port=8000

### step 3: Terminal 3 – Start Desktop App
cd frontend
npm run dev
