###################################################################
# Attention is all you need implementation inspired by Andrej 
# Karpathy's tutorial
###################################################################
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.nn import functional as F

class Head(nn.Module):
    # Implements single self attention head 
    def __init__(self, head_size):
        super().__init__()
        self.K = nn.Linear(n_embed, head_size, bias=False)          # key
        self.Q = nn.Linear(n_embed, head_size, bias=False)          # query
        self.V = nn.Linear(n_embed, head_size, bias=False)          # value
        self.triangular_matrix('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, time, channel = x.shape
        k = self.K(x)
        q = self.Q(x)
        weights = q @ k.transpose(-2,-1)* channel**-0.5
        weights = weights.masked_fill(self.tril[:time,:time]==0, float('-inf'))
        weights = F.softmax(weights, dim=1)
        weights = self.dropout(weights)
        v = self.V(x)
        return weights @ v

class MultuHeadSelfAttention(nn.Module):
    # Calls Head class in parallel for faster computation
    def __init__(self, number_of_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size=head_size) for i in range(number_of_heads)])
        self.projection = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=1)
        return self.dropout(self.projection(out))
    

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embed // n_head
        self.self_attention = MultuHeadSelfAttention(n_head, head_size)
        # Linear layer to introduce non linearity
        self.ffwd_net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        x = x + self.self_attention(self.ln1(x))
        x = x + self.ffwd_net(self.ln2(x))
        return x

class Bigram(nn.Module):
    def __init__(self):
        super().__init__()
        """
        - Create embedding table
        - positional embedding
        """
        self.embedding_table_token = nn.Embedding(vocab_size, vocab_size) # Creates a table of Batch x Time x Channel shape
        self.positional_embedding = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for i in range(n_layer)])
        self.final_layer = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)
    
    def forward(self, idx, targets=None):
        # logits can predict what comes next. But the individual tokens are not communicating with each other
        B, T = idx.shape
        #logits = self.embedding_table_token(idx) 
        token_embedding = self.embedding_table_token(idx)
        positional_embed = self.positional_embedding(torch.arange(T, device=device)) 
        x = token_embedding + positional_embed
        x = self.blocks(x)
        x = self.final_layer(x)
        logits = self.head(x) 

        if targets == None:
            loss = None
        else:
            # Unpacking logits
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_tokens):
        """
        - idx is an array of size (B,T) 
        - steps to generate test:
            - Get predicttions
            - get the last prediction logits
            - apply softmax to get the probabilities
            - sample from distribution
            - append the sampled prediction to the running sequence
        """
        for i in range(max_tokens):
            logits, loss = self(idx)
            logits = logits[:,-1,:] #(BxC) matrix
            probabilities = F.softmax(logits, dim=1) # (BxC) matrix
            next_sequence = torch.multinomial(probabilities, num_samples=1) #(Bx1) matrix
            idx = torch.cat((idx, next_sequence), dim=1) # (BxT+1) matrix
        return idx

@torch.no_grad
def calculate_loss():
    res = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        res[split] = losses.mean()
    model.train()
    return res

# hyperparameters
batch_size = 16 
block_size = 32
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2    
device = 'cpu' # 'cuda' if torch.cuda.is_available() else
eval_iters = 200
n_embed = 32
n_head = 4
n_layer = 4
dropout = 0.0

torch.manual_seed(1337)

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

model = Bigram()
m = model.to(device)

# Number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = calculate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
