from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import torch
import os

# Explicitly disable tokenizer parallelism to avoid deadlock warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

max_length = 64
token_delimiter = ">>|[]|>>"
token_start = "<|paraphrase|>"

if __name__ == "__main__":
    def load_data(input_file, target_file):

        with open(input_file, "r", encoding="utf-8") as f:
            inputs = f.read().split("\n")
        with open(target_file, "r", encoding="utf-8") as f:
            targets = f.read().split("\n")
        assert len(inputs) == len(targets), "The number of inputs and targets do not match"

        # remove lines that exceed our limit based off words
        inputs, targets = zip(*[(i, t) for i, t in zip(inputs, targets) if len(i.split()) < max_length and len(t.split()) < max_length])

        return inputs, targets

    input_data, target_data = load_data("datasets/scrambled.txt", "datasets/target.txt")

    class Paraphrase():
        def __init__(self, control_code, input_data, target_data, truncate=True, gpt2_type="gpt2"):
            self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            identifying_sequence = token_delimiter

            self.input_data = []

            for i in range(len(input_data)):
                original_text = input_data[i]
                target_text = target_data[i]

                tokenized_input = self.tokenizer.encode(f"{control_code} {original_text} {identifying_sequence} {target_text}", return_tensors="pt", truncation=truncate, max_length=max_length)

                input_pad_length = max_length - tokenized_input.shape[1]

                if input_pad_length > 0:
                    # pad with eos tokens
                    input_padding = torch.tensor([self.tokenizer.eos_token_id] * input_pad_length).unsqueeze(0)
                    tokenized_input = torch.cat([tokenized_input, input_padding], dim=1)

                self.input_data.append(tokenized_input)
            
        def __len__(self):
            return len(self.input_data)
        
        def __getitem__(self, idx):
            return self.input_data[idx]

    # create our dataset object
    dataset = Paraphrase(token_start, input_data, target_data)

    # load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # accumualte batch size for faster training
    def pack_tensor(new_tensor, packed_tensor, max_seq_len):
        if packed_tensor is None:
            return new_tensor, True, None
        if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
            return packed_tensor, False, new_tensor
        else:
            packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
            return packed_tensor, True, None
    
    def custom_loss_function(outputs, labels, tokenizer, incentive_threshold=4, penalty=0.01):
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            predictions = shift_logits.view(-1, shift_logits.size(-1))

            predicted_words = [tokenizer.decode([tok]) for tok in torch.argmax(predictions, dim=1)]

            # filter out the stop tokens
            predicted_words = list(filter(lambda tok: tok not in [tokenizer.eos_token, tokenizer.pad_token], predicted_words))

            # incentivize longer words by scaling linearly
            total_reward = 0.0
            for word in predicted_words:
                if len(word) > incentive_threshold:
                    total_reward += penalty * len(word)

            # pass through default cross entropy loss
            loss = outputs.loss

            adjusted_loss = loss + total_reward

            return adjusted_loss
        
    # create the train function
    def train(dataset, model, tokenizer, batch_size=8, epochs=2, lr=0.001, max_seq_len=400, warmup_steps = 200, gpt2_top="gpt2", 
            output_dir=".", output_prefix="wreckgar", test_mode=False, save_model_on_epoch=False):

        device=torch.device("cuda")
        model = model.cuda()
        model.train()

        print("starting training process")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1)

        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loss = 0.0
        accumulating_batch_count = 0
        input_tensor = None

        for epoch in range(epochs):
            total_batches = len(train_dataloader)

            print(f"training epoch {epoch + 1}")

            # make one progress bar for each epoch
            progress_bar = tqdm(total=total_batches, desc=f"epoch {epoch + 1}", position=0, leave=True)
            progress_bar = tqdm(total=total_batches, desc=f"epoch {epoch + 1}", position=0, leave=True)

            for idx, entry in enumerate(train_dataloader):
                (input_tensor, can_pack, remainder) = pack_tensor(entry, input_tensor, max_seq_len)
                if not can_pack and idx != len(train_dataloader) - 1:
                    continue
                
                input_tensor = input_tensor.to(device)
                outputs = model(input_tensor, labels=input_tensor)
                loss = custom_loss_function(outputs, input_tensor, tokenizer)
                loss.backward()

                if (accumulating_batch_count % batch_size) == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    input_tensor = None
                    model.zero_grad()
                
                accumulating_batch_count += 1
                input_tensor = None

                progress_bar.update()
                progress_bar.set_description(f"epoch {epoch + 1} iter {idx + 1}/{total_batches} loss {loss.item():.2f}")

            if save_model_on_epoch:
                torch.save(model.state_dict(), f"{output_dir}/{output_prefix}_epoch_{epoch}.pt")
            
        return model

    model = train(dataset, model, tokenizer)

    # save the model
    model.save_pretrained("model1")

    # save the tokenizer
    tokenizer.save_pretrained("model1")

    def getModel():
        model = GPT2LMHeadModel.from_pretrained("model1")
        tokenizer = GPT2Tokenizer.from_pretrained("model1")
        return model, tokenizer


    def generate(prompt):
        model, tokenizer = getModel()
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

    # print(generate("ratification 13th abolished slavery act."))
    # print(generate("foxtrot uniform charlie kilo."))
    # print(generate("I dropped my sister off at school today, and then I went to the gym."))
