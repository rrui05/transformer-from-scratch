import torch.nn as nn
from einops import rearrange

class Multihead_self_attention(nn.MOdule):
    def __init__(self,heads,head_dim,dim):
        super().__init__()
        self.head_dim = head_dim # 每一个注意力头的维度
        self.heads = heads  # 注意力头个数
        self.inner_dim = self.heads * self.head_dim 
        self.scale = self.head_dim**-0.5  # 正则化系数
        self.to_qkv = nn.Linear(dim,self.inner_dim*3)  # 一次性生成QKV
        self.to_output = nn.Linear(self.inner_dim,dim)
        self.norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self,x):
        x = self.norm(x) # PreNorm
        qkv = self.to_qkv(x).chunk(3,dim=-1) 
        Q,K,V = map(lambda t: rearrange(t,'b l (h dim) -> b h l dim',dim=self.head_dim),qkv)
        K_T = K.transpose(-1,-2)
        att_score = Q@K_T*self.scale
        att = self.softmax(att_score)
        out = att@V  # (B,H,L,dim)
        out = rearrange(out,'b h l dim -> b l (h dim)') # 拼接
        output = self.to_output(out)
        return output
    
        