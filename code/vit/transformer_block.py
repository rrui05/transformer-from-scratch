from torch import nn
from MHA import Multihead_self_attention
from ffn import FeedForward

class Transformer_block(nn.Module):
    def __init__(self,dim,heads,head_dim,mlp_dim):
        super().__init__()
        self.MHA = Multihead_self_attention(heads = heads, head_dim = head_dim, dim = dim )
        self.FeedForward = FeedForward(dim=dim, mlp_dim=mlp_dim)


    def forward(self,x):
        x = self.MHA(x)+x
        x = self.FeedForward(x)+x
        return x

class Transformer(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, depth):
        super().__init__()
        self.layers = nn.Sequential(
            *(Transformer_block(dim, heads, head_dim, mlp_dim) for _ in range(depth))
        )

    def forward(self, x):
        return self.layers(x)
