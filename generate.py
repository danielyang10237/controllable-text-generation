from transformers import GPT2LMHeadModel, GPT2Tokenizer

from model import token_start, token_delimiter, max_length

model = GPT2LMHeadModel.from_pretrained("model1")
tokenizer = GPT2Tokenizer.from_pretrained("model1")

def generate(prompt):
    model.eval()

    prompt_input = tokenizer.encode(
        f"{token_start} {prompt} {token_delimiter}",
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    )

    generated_output = model.generate(
        prompt_input,
        max_length=150,
        num_beams=5,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        do_sample=True,
        pad_token_id=50256,
        early_stopping=True,
    )

    return tokenizer.decode(generated_output[0], skip_special_tokens=True)


print(generate("ratification 13th abolished slavery act."))
