class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff) # 第一个全连接层，将d_model维度映射到d_ff维度
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff,d_model)
        self.activation = nn.RelU()

    def forward(self,x):
        # x: (batch_size, seq_len, d_model)
        x = self.linear1(x) # (batch_size,seq_len,d_ff)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x) # (batch_size,seq_len,d_model)
        return x # (batch_size,seq_len,d_model)