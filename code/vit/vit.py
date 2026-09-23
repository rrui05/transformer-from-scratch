


class ViT(nn.Module):
    def __init__(self,image_size,channels,patch_size ,dim,heads,head_dim,mlp_dim,depth,num_class):
        super().__init__()
        self.to_patch_embedding = pre_proces(image_size = image_size,patch_size = patch_size, patch_dim=channels*patch_size**2, dim=dim)
        self.transformer = Transformer(dim = dim,heads=heads,head_dim= head_dim,mlp_dim=mlp_dim ,depth=depth)
        self.MLP_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,num_class)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        token = self.to_patch_embedding(x)
        output = self.transformer(token)
        CLS_token = output[:,0,:]
        out = self.softmax(self.MLP_head(CLS_token)) # 注意最后的分类是基于CLS token做的
        return out