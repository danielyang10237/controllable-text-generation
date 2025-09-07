# import hugging face gpt j model
from transformers import GPTJForCausalLM, AutoTokenizer
import torch
import os
import math

# Explicitly disable tokenizer parallelism to avoid deadlock warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

max_length = 64
token_delimiter = ">>|[]|>>"
token_start = "<|<|&|>|>"

if __name__ == "__main__":

    def load_data(input_file, target_file):

        with open(input_file, "r", encoding="utf-8") as f:
            inputs = f.read().split("\n")
        with open(target_file, "r", encoding="utf-8") as f:
            targets = f.read().split("\n")
        assert len(inputs) == len(
            targets
        ), "The number of inputs and targets do not match"

        # remove lines that exceed our limit based off words
        inputs, targets = zip(
            *[
                (i, t)
                for i, t in zip(inputs, targets)
                if len(i.split()) < max_length and len(t.split()) < max_length
            ]
        )

        return inputs, targets

    input_data, target_data = load_data("datasets/scrambled.txt", "datasets/target.txt")

    class ParaphraseDataset():
        def __init__(self, control_code, input_data, target_data, truncate=True):
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
            self.tokenizer.pad_token = self.tokenizer.eos_token

            identifying_sequence = token_delimiter

            self.input_data = []

            for i in range(len(input_data)):
                original_text = input_data[i]
                target_text = target_data[i]

                tokenized_input = self.tokenizer.encode(
                    f"{control_code} {original_text} {identifying_sequence} {target_text}",
                    return_tensors="pt",
                    truncation=truncate,
                    max_length=max_length,
                )

                input_pad_length = max_length - tokenized_input.shape[1]

                if input_pad_length > 0:
                    # pad with eos tokens
                    input_padding = torch.tensor(
                        [self.tokenizer.eos_token_id] * input_pad_length
                    ).unsqueeze(0)
                    tokenized_input = torch.cat([tokenized_input, input_padding], dim=1)

                self.input_data.append(tokenized_input)

                if i % 5000 == 0:
                    print(f"processed {i} out of {len(input_data)}")

    device = "cuda"
    model = GPTJForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B",
        revision="float16",
        torch_dtype=torch.float16,
        pad_token_id=50256,
    ).to(device)

    finetune_dataset = ParaphraseDataset(token_start, input_data, target_data)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    def train(dataset, model, tokenizer, device, batch_size=16, epochs=5, lr=0.0001, warmup_steps = 200, test_mode=false):
        model.train()

        print("starting training process")

        model.zero_grad()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(dataset), epochs=epochs, anneal_strategy='linear', warmup_steps=warmup_steps)
        losses = []
        for epoch in range(epochs):
            for i in range(0, len(dataset), batch_size):
                batch = torch.stack(dataset.input_data[i:i+batch_size]).to(device)
                outputs = model(input_ids=batch, labels=batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                losses.append(loss.item())
                if i % 5000 == 0:
                    print(f"processed {i} out of {len(dataset)}")
                    print(f"epoch {epoch} loss: {loss.item()}")
                if test_mode:
                    break
        return losses
