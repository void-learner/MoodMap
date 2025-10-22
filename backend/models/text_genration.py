from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import spacy

nlp = spacy.load("en_core_web_sm")

# model
model_name = "microsoft/DialoGPT-medium"
device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True).to(device)

# for natural and engaging conversations
filters = ['hmm', 'you know', 'I see', 'oh', 'well', 'like', 'so']
def add_filters(text, prob=0.2):
    if random.random() < prob:
        filter = random.choice(filters)
        text = f"{filter}, {text}"
        return text
    return text

def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

def chat_with_bot(emotion, user_input, user_id='User'):
    # if history is None:
    #     history = []

    emotion_label = [e['label'] for e in emotion]
    name = extract_name(user_input) or user_id


    prompt = "You are a friendly and empathetic chatbot named Goru. Speak naturally and emotionally based on the user's feelings."
    top_emotions = emotion_label[:2]
    prompt += f" Given emotions: {', '.join(top_emotions)}."

    # emotional tone adjustment
    emotional_tone_map = {
        'joy': "cheerful",
        'amusement': "cheerful",
        'excitement': "cheerful",
        'sadness': "empathetic",
        'grief': "empathetic",
        'disappointment': "empathetic",
        'remorse': "empathetic",
        'anger': "calm",
        'annoyance': "calm",
        'disapproval': "calm",
        'disgust': "calm",
        'fear': "reassuring",
        'nervousness': "reassuring",
        'love': "warm",
        'caring': "warm",
        'admiration': "warm",
        'confusion': "helpful",
        'realization': "helpful",
        'curiosity': "friendly",
        'desire': "friendly",
        'approval': "positive",
        'gratitude': "positive",
        'relief': "positive",
        'optimism': "positive",
        'pride': "positive",
        'embarrassment': "gentle",
        'surprise': "curious",
        'none': "neutral"
    }


    for e in top_emotions:
        if e in emotional_tone_map:
            prompt += " " + emotional_tone_map[e]

    # check for conversation history
    # if history:
    #     prompt += " Previous conversation:\n"
    #     for msg in history[-5:]:
    #         prompt += f"User: {msg['role']}\nAI: {msg['content']}\n"

    # Add current user input
    prompt += f"\n{name}: {user_input}\nAI:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=512,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,    # Creative, varied text generation
            num_return_sequences=1,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=5
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = response.split("Goru:")[-1].strip()
    response = add_filters(response, prob=0.2)

    return response
    
# examples
print(chat_with_bot([{'label': 'joy'}], "Hey Goru! I got selected for my internship!"))
print(chat_with_bot([{'label': 'sadness'}], "I miss someone who doesn’t talk to me anymore."))
print(chat_with_bot([{'label': 'curiosity'}], "Why do people overthink so much, Goru?"))
print(chat_with_bot([{'label': 'anger'}], "I'm so annoyed, my project group never listens to my ideas!"))
print(chat_with_bot([{'label': 'love'}], "I really admire someone, Goru. They make everything feel peaceful."))
print(chat_with_bot([{'label': 'fear'}], "I have my presentation tomorrow, I’m so anxious."))
print(chat_with_bot([{'label': 'confusion'}], "I'm not sure if I chose the right path in life."))
print(chat_with_bot([{'label': 'gratitude'}], "Thanks for listening to me, Goru."))
print(chat_with_bot([{'label': 'optimism'}], "I feel like things are finally starting to make sense now."))
print(chat_with_bot([{'label': 'embarrassment'}], "I tripped in front of everyone today."))
print(chat_with_bot([{'label': 'none'}], "What’s your favorite thing about conversations, Goru?"))
print(chat_with_bot([{'label': 'love'}], "Sometimes I feel emotions too deeply. It’s beautiful but exhausting."))



# # Choose the model
# model_name = "facebook/blenderbot-1B-distill"
# tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
# model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# def chat_with_bot(emotion, user_input):
#     emotion_labels = [e['label'] for e in emotion]
#     prompt = f"Given emotions: {', '.join(emotion_labels)}. Respond cheerfully if happy, empathetically if sad: {user_input}"
#     inputs = tokenizer(
#         prompt, 
#         return_tensors="pt",
#         truncation=True
#     )
    
#     reply_ids = model.generate(
#         **inputs,
#         max_length=512,
#         temperature=0.7,
#         top_k=50,
#         top_p=0.9,
#         do_sample=True,    # Creative, varied text generation
#         repetition_penalty=1.2,
#         num_beams=5
#     )

#     return tokenizer.decode(reply_ids[0], skip_special_tokens=True)

# examples
# print("Bot:", chat_with_bot("happy", "Hello, how are you?. My name is Aarya"))
# print("Bot:", chat_with_bot("neutral", "I love poetry."))



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
