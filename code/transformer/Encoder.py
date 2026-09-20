'''
基于MHA和FFN
'''
class EncoderLayer(nn.Module):
    def __init__(self,d_model,n_heads,d_ff,dropout=0.1):
        super().__init__()
        # self attention: 自注意力机制,QKV都来自同一个序列,用于捕捉序列中不同位置之间的依赖关系，比如it在句子中指代什么
        self.self_attn = MultiHeadAttention(d_model,n_heads,dropout) # 多头自注意力机制
        self.dropout1 == nn.Dropout(dropout)
        # LayerNorm: 对最后一个维度归一化，也就是单取最后一个维度的时候看到的是均值为0，方差为1的分布，LayerNorm是对每个样本独立归一化的
        # 每个token都是横着放的，横着的那个维度
        self.norm1 = nn.LayerNorm(d_model) 

        self.ffn = FeedForward(d_model,d_ff,dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)