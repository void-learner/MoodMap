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

# import tiktoken  # have to install it via pip

#gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
# gpt2_tokenizer =  GPT2Tokenizer.from_pretrained("openai-community/gpt2")
# oai_tokenizer = tiktoken.get_encoding("gpt2")

# orig = "Is this restaurant family-friendly ? Yes No Unsure ? This is an other sentence ."

# hf_enc = gpt2_tokenizer(orig)["input_ids"]
# hf_dec = gpt2_tokenizer.decode(hf_enc)

# oai_enc = oai_tokenizer.encode(orig)
# oai_dec = oai_tokenizer.decode(oai_enc)

# print(hf_dec)
# print(oai_dec)


# from transformers import pipeline

# # Load a pre-trained text generation pipeline
# generator = pipeline("text-generation", model="gpt2")

# # Input prompt for text generation
# prompt = "I am very sad today because I got into an argument with my best friend."

# # Generate text
# results = generator(
#     prompt,
#     max_length=400,        
#     do_sample=True, 
#     top_k=50, 
#     top_p=0.95, 
#     temperature=0.8,
#     repetition_penalty=1.2
# )

# # Print generated text
# for result in results:
#     print(result['generated_text'])


# import os
# import openai
# openai.api_key = "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"
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

from transformers import pipeline
from transformers import Conversation

# Load a pre-trained conversational model and tokenizer
chatbot = pipeline("conversational", model="facebook/blenderbot-400M-distill")
# Create a conversation object with the user's first input
conversation = Conversation("Hello, who are you?")

# Pass the conversation object to the chatbot pipeline
conversation = chatbot(conversation)

# The conversation object now contains the full chat history
print(conversation)