'''
预处理部分，包含：
1.将Image切分成patches
2.每个patch映射成token，一个patch就是一个token
3.加入CLS token
4.加入position embedding

最终的输出可以直接送入transformer encoder、

为了便于展示，这里默认输入为正方形
'''

import torch 
from torch import nn
from einops import rearrange, repeat

class pre_process(nn.Module):
    # image_size: 输入图片的边长
    # patch_size: patch的边长
    # patch_dim: 一个patch展平后的维度，通常是patch_dim = patch_size^2 × channels
    # dim: transformer里每个token的embedding维度

    def __init__(self,image_size,patch_size,patch_dim,dim):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.patch_num = (image_size//paich_size)**2
        self.linear_embedding = nn.Linear(patch_dim,dim) # 把patch投影成transformer 要求的token维度，一个全连接解决
        # patch_token经过linear_embedding后，shape是(B,L,C)
        # B: batch_size
        # L: token(patch)数量
        # C: 每个token(patch)的embedding维度，也即是dim
        self.position_embedding = nn.Parameter(torch.randn(1,self.patch_num+1,self.dim)) # 注意这里的位置编码是要参与训练更新的，+1是因为由CLS token
        # 注意位置编码和transform不一样，这里的位置编码是可以学习得到的。
        # CLS token是用来汇总整张图片信息的token，再transformer每一层中会让CLS对其他patch token做attention，得以逐渐聚合全图信息
        # CLS token也是要参与更新的
        self.CLS_token = nn.Parameter(torch.randn(1,1,self.dim)) # 后续会广播 
        
    def forward(self,x):
        # 输入原图x: (B, C, H, W)，将图片切成patch
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)  # (B,L,C)
        x = self.linear_embedding(x)
        b,l,c = x.shape 
        CLS_token = repeat(self.CLS_token,'1 1 d -> b 1 d',b=b) # 位置编码复制B份
        x = torch.concat((CLS_token,x), dim=1)
        x = x + self.position_embedding
        return x
        
        
         

    