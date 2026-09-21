class DecoderLayer(nn.Module):
    def __init__():
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model,n_heads,dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.Layernorm(d_model)
        
        self.cross_attn = MultiHeadAttention(d_model,n_heads,dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.Layernorm(d_model)
        
        self.ffn  = FeedForward(d_model,n_heads,dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm3 = nn.Layernorm(d_model)
        
        
    def forward(self,tgt,src,tgt_mask = None ,src_mask = None):
        # tgt: (batch_size, tgt_seq_len, d_model)
        # memory: (batch_size, src_seq_len, d_model)
        # tgt_mask: (batch_size, 1, 1, tgt_seq_len) Decoder 当前处理的目标序列
        # src_mask: (batch_size, 1, 1, src_seq_len) Encoder 的输出
        
        x = tgt 
        output = self.self_attn(x,x,x,tgt_mask) # pytorch 自动调用forward函数
        x = self.norm1(x + self.dropout(output))
        
        output = self.cross_attn(x,src,src,src_mask) # kv和Q的seq_len是可以不一样的，因为最后是Q乘K的转置，特征维度（d_model）一样就可以
        x = self.norm2(x + self.dropout2(output))
        
        output = self.ffn(x)
        x = self.norm3(x + self.dropout3(output))
        return x
    
class Decoder(nn.Module):
    def __init__(self,d_model,n_heads,d_ff,num_layers,dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model,d_heads)])
        
    def forward(self,x,memory,tgt_mask = None,memory_mask = None):
        # x:(batch_size,seq_len,d_model)
        for layer in self.layers:
            x = layer(x,memory,tgt_mask,memory_mask) # x:(batch_size,seq_len,d_model)
        return x   # x:(batch_size,seq_len,d_model)
        
        
        
        
        
        