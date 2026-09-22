import torch 
from torch import nn

class pre_process(nn.Module):
    def __init__(self,image_size,patch_size,patch_dim,dim):

