from transformers import GPT2LMHeadModel, GPT2Tokenizer

from model import token_start, token_delimiter, max_length, default_train, make_longer_train

import nltk
if not nltk.download('stopwords'):
    nltk.download('stopwords')
import random

from nltk.corpus import stopwords
english_stopwords = set(stopwords.words('english'))

import random
import torch

default_gen = True
make_longer_gen = True

model_type = "model1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPT2LMHeadModel.from_pretrained(model_type).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_type)

# prompt_dest = 'paraphrased-text/prompts.txt'
# generated_dest = 'paraphrased-text/generated.txt'

prompt_dest = 'gen-outputs/prompts.txt'
generated_dest = 'gen-outputs/generated.txt'

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

    generated_output = prompt_input
    
    def sample_longer_tokens(logits, top_k=50, length_factor = 0.02, temperature=0.7, num_beams=3):
        # next_token_id = torch.multinomial(probs, num_samples=1).unsqueeze(0)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # use beam search to determine the next token
        top_k_tokens_indexes = torch.multinomial(probs, num_samples=top_k)

        # create a tensor length top_k
        scaled_prob_tensor = torch.zeros(top_k)

        # set the scaled_prob_tensor to the probability of the top_k tokens
        for i in range(top_k):
            new_prob = probs[top_k_tokens_indexes[i]]
            num_letters = len(tokenizer.decode(top_k_tokens_indexes[i].unsqueeze(0), skip_special_tokens=True))
            if make_longer_gen:
                scaled_prob_tensor[i] = new_prob * (1 + length_factor * num_letters)
            else:
                scaled_prob_tensor[i] = new_prob * (1 - length_factor * num_letters)
            # scale the stop token too
            if top_k_tokens_indexes[i] == 50256:
                scaled_prob_tensor[i] = new_prob * (1 - length_factor * 4)
        
        # randomly sample from the top_k tokens with temperature implementation
        scaled_prob_tensor = scaled_prob_tensor / temperature
        scaled_prob_tensor = torch.nn.functional.softmax(scaled_prob_tensor, dim=-1)
        next_token_id = top_k_tokens_indexes[torch.multinomial(scaled_prob_tensor, num_samples=2)]

        if generated_output.shape[1] > 0 and next_token_id[0] == generated_output[0, -1] and next_token_id[1] != 50256:
            next_token = next_token_id[1]
            return next_token.unsqueeze(0).unsqueeze(0)

        next_token = next_token_id[0]
        return next_token.unsqueeze(0).unsqueeze(0)

    if default_gen:
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
    else:
        with torch.no_grad():
            for i in range(max_length):
                outputs = model(generated_output, labels=generated_output)
                logits = outputs.logits
                next_token_logits = logits[0, -1, :]
                next_token_id = sample_longer_tokens(next_token_logits)
                generated_output = torch.cat((generated_output, next_token_id), dim=1)
                if next_token_id == 50256:
                    break
                if i == max_length - 1:
                    break

    return tokenizer.decode(generated_output[0], skip_special_tokens=True)

with open(prompt_dest, 'r') as prompts, open(generated_dest, 'w') as generated:
    for prompt in prompts:
        prompt = prompt.strip()
        if prompt:
            generated_line = generate(prompt)
            index_token_delimiter = generated_line.find(token_delimiter)
            print("PROMPT:", generated_line[len(token_start):index_token_delimiter])
            print("GENERATED:", generated_line[index_token_delimiter + len(token_delimiter):])
            generated.write(generated_line[index_token_delimiter + len(token_delimiter):] + '\n')

print("Done generating!")