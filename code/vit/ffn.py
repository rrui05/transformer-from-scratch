from torch import nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self,dim,mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim,mlp_dim)
        self.fc2 = nn.Linear(mlp_dim,dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self,x):
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x
    
        