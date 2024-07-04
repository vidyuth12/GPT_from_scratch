# importing the required libraries
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

# hyperparameters
batch_size = 32 
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32

# read the data we will be training on
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

"""
Creating a vocabulary of all the characters present in the 
input text. Also, creating an encoder and a decoder using a 
simple dictionary.
"""
chars = sorted(list(set(text)))              # Our vocabulary
vocab_size = len(chars)                      # Size of our vocabulary
# print(''.join(chars))
# print(vocab_size)

# code for tokenising input text
a_z = {ch:i for i, ch in enumerate(chars)}
z_a = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [a_z[c] for c in s]          # Encoder: takes a string and outputs a list of integers
decode = lambda l: ''.join([z_a[i] for i in l]) # Decoder: takes a list of integer, and outputs a string

# print(encode("hii there"))
# print(decode(encode("hii there")))

# Tokenising the input text
data = torch.tensor(encode(text), dtype=torch.long)

# Splitting the dataset to train and test
n = int(0.9*len(data))
train_data = data[:n]
val_data  = data[n:]

# Loading data
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Implementing simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size) # (B,T,vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        token_embed = self.token_embedding_table(idx) # (B,T,C)
        pos_embed = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = token_embed + pos_embed
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# Driver code
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f" step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate text from model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
