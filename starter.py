
import torch

import os
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import keras_core as keras


from dataclasses import dataclass

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda x: "".join([itos[i] for i in x])



### Hyper-parameters
@dataclass
class Config:
    block_size = 64
    d_model = 512
    sample_size = 500
    vocab_size = len(chars)
    num_heads = 1
    key_dim = d_model
    batch_size = 32


class DataPrep():
    def __init__(self, config):
        self.enc_text = encode(text)
        self.block_size = config.block_size
        self.sample_size = config.sample_size

    def prepare(self):
        data = []
        ix = np.random.randint(0, len(text)-self.block_size, self.sample_size)
        for i in ix:
            data.append(self.enc_text[i:i+self.block_size])

        return torch.tensor(data).to("mps")
    
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        x = self.data[idx][:-1]
        y = self.data[idx][1:]

        return x, y

    def __len__(self):
        return len(self.data)


class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.embedding = keras.layers.Embedding(config.vocab_size, config.d_model)
        self.pe_embedding = keras.layers.Embedding(config.block_size, config.d_model)

    def call(self, x):
        b, t = x.size()
        pos = keras.ops.arange(0, t, dtype="int32")
        tok_emb = self.embedding(x)
        pos_emb = self.pe_embedding(pos)

        return tok_emb + pos_emb
    
class BaseAttention(keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.mha = keras.layers.MultiHeadAttention(config.num_heads, config.key_dim)
        self.layernorm = keras.layers.LayerNormalization()
        self.add = keras.layers.Add()

class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask = True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

class MyModel(keras.Model):
    def __init__(self, config):
        super().__init__()
        self.emb = PositionalEmbedding(config)
        self.csa = CausalSelfAttention(config)
        self.final = keras.layers.Dense(config.vocab_size)

    def call(self, x):
        x = self.emb(x)
        x = self.csa(x)
        logits = self.final(x)

        return logits


    
data = DataPrep(Config).prepare()
ds = MyDataset(data)
dl = torch.utils.data.DataLoader(ds, batch_size=Config.batch_size)

model = MyModel(Config).to("mps")

optimizer = keras.optimizers.AdamW()
loss = keras.losses.SparseCategoricalCrossentropy()
    
model.compile(
    optimizer=optimizer,
    loss=loss
)

model.fit(dl, epochs=10000)

model.save("final_model.keras")
model = keras.saving.load_model("final_model.keras")


    
