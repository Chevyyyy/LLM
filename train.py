import torch
from transformer_scratch import *

from datasets import load_dataset


model = GPTLanguageModel(vocab_size).to(device)
try:
    model.load_state_dict(torch.load("model.pt"))
    print("model loaded")
except:
    print("model not found: using new model")

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("strat training")
for iter in range(max_iter):
    if iter % eval_intervel == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss {losses['train']}, val loss {losses['val']}")
        torch.save(model.state_dict(), "model.pt")

    xb, yb = get_batch("train")

    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "model.pt")
print("model saved")