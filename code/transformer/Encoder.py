'''
EncoderLayer 基于MHA和FFN, Encoder层由若干个EncoderLayer组成
'''
class EncoderLayer(nn.Module):
    def __init__(self,d_model,n_heads,d_ff,dropout=0.1):
        super().__init__()
        # self attention: 自注意力机制,QKV都来自同一个序列,用于捕捉序列中不同位置之间的依赖关系，比如it在句子中指代什么
        self.self_attn = MultiHeadAttention(d_model,n_heads,dropout) # 多头自注意力机制
        self.dropout1 = nn.Dropout(dropout)
        # LayerNorm: 对最后一个维度归一化，也就是单取最后一个维度的时候看到的是均值为0，方差为1的分布，LayerNorm是对每个样本独立归一化的
        # LayerNorm输入参数必须对应最后n个维度
        # 每个token都是横着放的，横着的那个维度
        self.norm1 = nn.LayerNorm(d_model) 

        self.ffn = FeedForward(d_model,d_ff,dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self,x,mask=None):
        # x:(batch_size,seq_len,d_model)
        attn_output = self.self_attn(x, x, x, mask) # (batch_size,seq_len,d_model)
        x = self.norm1(x + self.dropout1(attn_output)) # 残差连接+归一化
        ffn_output = self.fnn(x) # (batch_size,seq_len,d_model)
        x = self.norm2(x + self.dropout2(ffn_output)) 
        return x # (batch_size,seq_len,d_model)
    
class Encoder(nn.Module):
    def __init__(self,d_model,n_heads,d_ff,num_layer,dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model,n_heads,d_ff,dropout) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self,x,mask=None):
        # x:(batch_size,seq_len,d_model)
        for layer in self.layers:
            x = layer(x,mask) # (batch_size,seq_len,d_model)
        x = self.norm(x) # (batch_size,seq_len,d_model)
        
        return x # (batch_size,seq_len,d_model)
        
        
        