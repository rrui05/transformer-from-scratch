'''
预处理部分，包含：
1.将Image切分成patches
2.每个patch映射成token
3.加入CLS token
4.加入position embedding

最终的输出可以直接送入transformer encoder、

为了便于展示，这里默认输入为正方形
'''

import torch 
from torch import nn

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
        self.linear_embedding = nn.Linear(patch_dim,dim)

