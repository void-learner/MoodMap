import ollama
from backend.models.personal_memory import extract_and_save_user_info

conversation_history = {}

def chat_with_bot(user_input: str, emotions: list, session_id: str) -> str:
    print(f"\n[DEBUG] User said: {user_input}")
    print(f"[DEBUG] Session ID: {session_id}")
    
    try:
        profile = extract_and_save_user_info(session_id, user_input)
        name = profile.get("name", "there")
        print(f"[DEBUG] Profile loaded → Name: {name}, Age: {profile.get('age')}, Location: {profile.get('location')}")

        if session_id not in conversation_history:
            conversation_history[session_id] = []
        history = conversation_history[session_id]

        emotion_labels = [e["label"] for e in emotions]
        top_emotions = emotion_labels[:2] if emotion_labels else ["neutral"]
        print(f"[DEBUG] Top emotions: {top_emotions}")

        system_prompt = f"""You are Goru, a warm and caring friend talking to {name}.
        {f'{name} is {profile["age"]} years old.' if profile.get("age") else ''}
        {f'They live in {profile["location"]}.' if profile.get("location") else ''}
        Be {' and '.join(top_emotions)}. Use their name often. Respond naturally."""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history[-10:])
        messages.append({"role": "user", "content": user_input})

        print("[DEBUG] Calling Ollama with model: gemma2:2b")
        print(f"[DEBUG] Messages sent: {len(messages)} messages")

        # THIS IS THE MOMENT OF TRUTH
        response = ollama.chat(
            model="gemma2:2b",   # ← MUST BE EXACTLY THIS
            messages=messages,
            options={
                "num_gpu": 0,           # ← THIS LINE FORCES CPU ONLY
                "temperature": 0.8,
                "num_predict": 256
            }
        )

        reply = response["message"]["content"].strip()
        print(f"[SUCCESS] Goru replied: {reply}")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})

        return reply

    except Exception as e:
        error_msg = str(e)
        print(f"\n[OLLAMA CRASHED] → {error_msg}\n")
        
        # THESE ARE THE MOST COMMON ERRORS & FIXES:
        if "model" in error_msg and "not found" in error_msg:
            return "Bro, Ollama can't find 'gemma2:2b'. Run: ollama pull gemma2:2b"
        elif "connection" in error_msg.lower():
            return "Ollama server not running. Start it with: ollama serve"
        else:
            return f"Goru is having a brain freeze... Error: {error_msg}"

# conversation_history = {}

# def chat_with_bot(emotion, user_input: str, session_id: str) -> str:
#     profile = extract_and_save_user_info(session_id, user_input)
#     user_name = profile["name"] 

#     # get conversation history
#     if session_id not in conversation_history:
#         conversation_history[session_id] = []
#     history = conversation_history[session_id]

#     # extract top emotions
#     emotion_label = [e['label'] for e in emotion]
#     top_emotions = emotion_label[:2]

#     system_prompt = f"""You are a warm, caring friend talking to {user_name}.
#     {f'{user_name} is {profile["age"]} years old.' if profile["age"] else ''}
#     {f'They live in {profile["location"]}.' if profile["location"] else ''}
#     Remember what they’ve told you before."""

#     system_prompt += f" Given emotions: {', '.join(top_emotions)}."

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
#             system_prompt += " " + emotional_tone_map[e] 

#     # Prepare messages
#     messages = [{"role": "system", "content": system_prompt}]
#     messages.extend(history[-10:])
#     messages.append({"role": "user", "content": user_input})   

#     # Call ollama
#     response = ollama.chat(model="gemma2:2b", 
#                            messages=messages,
#                            options={
#                                "temperature": 0.7,
#                                "top_p": 0.9,
#                            })
#     bot_reply = response["message"]["content"]

#     # Save to conversation history
#     history.append({"role": "user", "content": user_input})
#     history.append({"role": "assistant", "content": bot_reply})
    
#     return bot_reply




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