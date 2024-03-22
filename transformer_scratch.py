import torch
import torch.nn as nn
import torch.nn.functional as F

with open("DT.txt", "r", encoding="utf-8") as f:
    text = f.read()
chars = sorted(set(text))
vocab_size = len(chars)

device = "cuda" if torch.cuda.is_available() else "cpu"
block_size =128
batch_size = 64 
max_iter = 30000 
eval_iter = 20 
eval_intervel=100
dropout = 0.2
n_embd = 384 
n_layer = 8
n_head = 8
learning_rate = 1e-4

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# two dicts for str<->int convert
string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}


encode = lambda s: [string_to_int[ch] for ch in s]
decode = lambda l: "".join([int_to_string[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)
# print(data[:100])

n = int(0.8 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    x.to(device)
    y.to(device)
    return x, y





class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # shirk the full n_embed space into head_size space by linear transformation
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # B: batch_size, T: block_size, C: n_embed
        k = self.key(x)  # B,T,hs
        q = self.query(x)  # B,T,hs

        wei = (
            q @ k.transpose(-1, -2) * k.shape[-1] ** -0.5
        )  # (B,T,hs)@(B,hs,T)->(B,T,T)

        # softmax will make -inf to 0, so we mask the lower triangle to -inf
        # without mask, then the x can direct assess to the future target -> meaning less for training
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B,T,T)

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)  # B,T,hs
        out = wei @ v  # (B,T,T)@(B,T,hs)->(B,T,hs)

        return out


class MutiHeadAttention(nn.Module):
    def __init__(self, num_head, head_size):
        super().__init__()
        # define each heads in the muti-head attention
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        # project the mutile heads' output to full n_embed space
        self.proj = nn.Linear(num_head * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # cat all the heads' output
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        # different head will capture different features
        head_size = n_embd // n_head
        self.sa = MutiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln_1 = nn.LayerNorm(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # resnet connection
        y = self.sa(x)
        x = self.ln_1(x + y)
        y = self.ffwd(x)
        x = self.ln_2(x + y)
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # embed the char into n_embed space
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # embed the position of the char into n_embed space
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self.__init__weights)

    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)

    def forward(self, index, targets=None):
        B,T=index.shape

        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T).to(device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    @torch.no_grad()
    def generate(self, index, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            index_block=index.clone().detach()
            if index.shape[1]>block_size:
                index_block=index[:,-block_size:]
            
                 
            logits, loss = self.forward(index_block)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            index_next = torch.multinomial(probs, num_samples=1)

            index = torch.cat((index, index_next), dim=1)
        self.train()
        return index


