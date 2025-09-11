from transformers import GPT2Tokenizer, GPT2LMHeadModel
# import tiktoken  # have to install it via pip

get_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
get_model = GPT2LMHeadModel.from_pretrained("gpt2")

def generate_text(emotion, input_text):
    prompt = f"Given emotions: {', '.join(emotion)}. Respond cheerfully if happy, empathetically if sad: {input_text}"
    inputs = get_tokenizer(prompt, return_tensors="pt")  # Converts your prompt into token IDs
    outputs = get_model.generate(
        inputs['input_ids'], 
        max_length=150, 
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=get_tokenizer.eos_token_id,  # To avoid warnings about no pad_token_id
        repetition_penalty=1.2, 
    )

    return get_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)


print(generate_text(["joy", "love"], "I just got a new job!"))


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
