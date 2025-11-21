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

        response = ollama.chat(
            model="gemma2:2b",   
            messages=messages,
            options={
                "num_gpu": 0,    
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

        if "model" in error_msg and "not found" in error_msg:
            return "Bro, Ollama can't find 'gemma2:2b'. Run: ollama pull gemma2:2b"
        elif "connection" in error_msg.lower():
            return "Ollama server not running. Start it with: ollama serve"
        else:
            return f"Goru is having a brain freeze... Error: {error_msg}"

