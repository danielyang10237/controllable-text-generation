from transformers import GPT2LMHeadModel, GPT2Tokenizer

from model import token_start, token_delimiter, max_length

import nltk
if not nltk.download('stopwords'):
    nltk.download('stopwords')
import random

from nltk.corpus import stopwords
english_stopwords = set(stopwords.words('english'))

import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPT2LMHeadModel.from_pretrained("model1").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("model1")

def generate(prompt):
    model.eval()

    word_sentence = prompt.split()
    # remove end period if there
    if word_sentence[-1][-1] == '.':
        word_sentence[-1] = word_sentence[-1][:-1]
    
    for i, word in enumerate(word_sentence):
        if word not in english_stopwords:
            if random.random() < 0.24:
                # shuffle the word
                j = random.randint(0, len(word_sentence) - 1)
                word_sentence[i], word_sentence[j] = word_sentence[j], word_sentence[i]
    
    # add back the period
    if word_sentence[-1][-1] != '.':
        word_sentence[-1] = word_sentence[-1] + '.'
    
    prompt = ' '.join(word_sentence)

    prompt_input = tokenizer.encode(
        f"{token_start} {prompt} {token_delimiter}",
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    ).to(device)

    generated_output = model.generate(
        prompt_input,
        max_length=max_length,
        num_beams=3,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=25,
        top_p=0.95,
        temperature=0.7,
        do_sample=True,
        pad_token_id=50256,
        early_stopping=True,
    )

    return tokenizer.decode(generated_output[0], skip_special_tokens=True)

with open('gen-outputs/prompts.txt', 'r') as prompts, open('gen-outputs/generated.txt', 'w') as generated:
    for prompt in prompts:
        prompt = prompt.strip()
        if prompt:
            generated_line = generate(prompt)
            # generated.write(generated_line + '\n')
            index_token_delimiter = generated_line.find(token_delimiter)
            print("PROMPT:", generated_line[len(token_start):index_token_delimiter])
            print("GENERATED:", generated_line[index_token_delimiter + len(token_delimiter):])

print("Done generating!")