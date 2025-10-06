from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Choose the model
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

def chat_with_bot(emotion, user_input):
    emotion_labels = [e['label'] for e in emotion]
    prompt = f"Given emotions: {', '.join(emotion_labels)}. Respond cheerfully if happy, empathetically if sad: {user_input}"
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        truncation=True
    )
    
    reply_ids = model.generate(
        **inputs,
        max_length=512,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        do_sample=True,    # Creative, varied text generation
        repetition_penalty=1.2,
        num_beams=5
    )

    return tokenizer.decode(reply_ids[0], skip_special_tokens=True)

# examples
# print("Bot:", chat_with_bot("happy", "Hello, how are you?. My name is Aarya"))
# print("Bot:", chat_with_bot("neutral", "I love poetry."))


# =======================================================================


# from transformers import GPT2Tokenizer, GPT2LMHeadModel

# get_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# get_model = GPT2LMHeadModel.from_pretrained("gpt2")

# def generate_text(emotion, input_text):
#     prompt = f"Given emotions: {', '.join(emotion)}. Respond cheerfully if happy, empathetically if sad: {input_text}"
#     inputs = get_tokenizer(prompt, return_tensors="pt")  # Converts your prompt into token IDs
#     outputs = get_model.generate(
#         inputs['input_ids'], 
#         max_length=150, 
#         num_return_sequences=1,
#         do_sample=True,
#         top_k=50,
#         top_p=0.95,
#         temperature=0.7,
#         pad_token_id=get_tokenizer.eos_token_id,  # To avoid warnings about no pad_token_id
#         repetition_penalty=1.2, 
#     )

#     return get_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)


# print(generate_text(["joy", "love"], "I just got a new job!"))

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
