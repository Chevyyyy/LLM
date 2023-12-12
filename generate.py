import torch
from transformer_scratch import *


device = "cuda" if torch.cuda.is_available() else "cpu"

with open("wizard_of_oz.txt", "r", encoding="utf-8") as f:
    text = f.read()
chars = sorted(set(text))
vocab_size = len(chars)

model = GPTLanguageModel(vocab_size).to(device)
model.load_state_dict(torch.load("model.pt"))
model.eval()


prompt = input("Enter a prompt:\n")
len_prompt = len(prompt)
context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
generated_chars = decode(model.generate(context, max_new_tokens=128)[0].tolist())
print("\nResponse:")
print(generated_chars[len_prompt:])

print("total parameters:", count_parameters(model)//1e6, "M")
