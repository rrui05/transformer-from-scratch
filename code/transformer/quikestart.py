'''
Build a standard Encoder-Decoder transformer,then make a batchh of random token input,simulate a forward pass through the model, and print the output shape.
'''

import torch
from transformer.model import Transformer
from transformer.mask import creat_padding_mask

model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    n_heads=8,
    d_ff=2048,  # Feed Forward Network hidden layer dimension
    numlayers=6
)
'''
src: source sequence
tgt: target sequence
e.g. src = "I love machine learning"
     tgt = "我 喜欢   机器    学习"
'''
# randint:生成一个形状为 (32, 10) 的随机整数张量，值在 [0, 10000) 范围内
src = torch.randint(0,10000,(32,10)) # batch_size=32, seq_len=10
tgt = torch.randint(0,10000,(32,12)) # batch_size=32, seq_len=12
# src.shape: (32, 10),表示随机生成32个样本，每个source有10个token，每个token的值在[0, 10000)范围内
src_mask,tgt_mask = creat_padding_mask(src,tgt)

logits = model(src,tgt,src_mask,tgt_mask)
print(logits.shape) # (batch_size, seq_len, vocab_size)


