import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataloader,TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import os


# 新建images 文件夹
os. makedirs('images',exits_ok=True)

def set_seed(seed = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

## ---正弦时间位置编码---- ##
# 将连续的(diffusion)时间步转化为高维向量，连时间变化都能建模
class SinusoidalPositionEmdeddings(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim = dim
        
    def forward(self,time):
        device = time.device
        half_dim = self.dim // 2
        # 时间转向量：
        
        # 这里的操作其实和transformer位置编码的思路很像，都是转化到不同频率的sin，cos
        # 首先我们生成不同频率，那这里为什么是dim//2呢？
        # 因为每个频率同时对应了sin和cos，维度就乘2了
        # 先生成dim/2个不同频率，然后用时间t乘这些频率，放到sin，cos里
        # 最后得到token_t: [sin(tw_0),cos(t_w0),sin(tw_1),cos(tw_1)......sin(tw_m-1),cos(tw_m-1)]
        
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)  # 生成频率base
        embeddings = torch.exp(torch.arange(half_dim,device=device)*-embeddings)  
        # 将时间t与base频率相乘
        embeddings = time[:,None]*embeddings[None,:]
        embeddings = torch.cat((embeddings.sin(),embeddings.cos())，dim=-1)
        return embeddings
    
    class DiTBlock(nn.Module):
        def __init__(self,hidden_size,num_heads,mlp_ratio=4.0)
            super().__init__()
            self.norm1 = nn.LayerNorm(hidden_size,elementwise_affine=False,eps=1e-6)
            self.attn = nn.MultiheadAttention(hidden_size,num_heads,batch_first=True)
            self.norm2 = nn.LayerNorm(hidden_size,elementwise_affine=False,eps=1e-6)
            







