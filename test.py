# import sqlite3
# print(sqlite3.sqlite_version)
# print(sqlite3.version)

# import torch
# print("CUDA Available:", torch.cuda.is_available())
# print("GPU Name:", torch.cuda.get_device_name(0))
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.version.cuda)


# from datasets import load_dataset
# dataset = load_dataset("go_emotions", "simplified")
# emotion_names = dataset['train'].features['labels'].feature.names
# print(emotion_names)

# test_bot.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))  # So it finds 'models' folder

from backend.models.text_genration import chat_with_bot
from backend.models.emotion_analyzer import analyze_emotion

# Fake session
session_id = "test_user_123"

print("Your Emotional AI Friend is LIVE")
print("Type 'quit' to exit\n")

while True:
    user_msg = input("You: ").strip()
    if user_msg.lower() in ["quit", "exit", "bye"]:
        print("Bot: Take care! I'm always here when you need me")
        break
    if not user_msg:
        continue

    # Get emotions (your old function)
    emotions = analyze_emotion(user_msg)

    # Generate reply
    reply = chat_with_bot(emotions, user_msg, session_id)

    print(f"Bot: {reply}\n")