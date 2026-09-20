class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_heads,dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads # 每个head的维度

        assert(
            self.d_k * n_heads == d_model
        ),f"d_model {d_model} not divisible by n_heads {n_heads}"  # 确保可以整除

        # nn.Linear:全连接层,输入维度(x)为d_model,输出维度(y)为d_model,bias=False表示不使用偏置项
        # y=xW^T+b(W^T是转置)
        # 特征数就是维度（描述一个对象所使用的数字个数）
        # Q=xW_Q
        # K=xW_K
        # V=xW_V
        self.W_q = nn.Linear(d_model,d_model,bias=False) # query的线性变换
        self.W_k = nn.Linear(d_model,d_model,bias=False) # key的线性变换
        self.W_v = nn.Linear(d_model,d_model,bias=False) # value的线性变换
        # 多个attention heads计算完成后对的结果会拼接
        # Output=Concat(head1​,…,headh​)W_O
        self.W_o = nn.Linear(d_model,d_model) # output的线性变换

        self.dropout = nn.Dropout(dropout) # dropout层,防止过拟合

    def forward(self,Q,K,V,mask=None):
        # Q: (batch_size, seq_len_q, d_model)
        # K: (batch_size, seq_len_k, d_model)
        # V: (batch_size, seq_len_v, d_model)

        batch_size = Q.size(0)

        # 将Q,K,V通过线性变换并拆分为多个head，.transpose(1, 2)，交换第一个和第二个维度，从0开始数
        Q = self.W_q(Q).view(batch_size,self.n_heads,-1,self.d_k).transpose(1,2) # (batch_size, n_heads, seq_len_q, d_k)
        K = self.W_k(K).view(batch_size,self.n_heads,-1,self.d_k).transpose(1,2) # (batch_size, n_heads, seq_len_k, d_k)
        V = self.W_v(V).view(batch_size,self.n_heads,-1,self.d_k).transpose(1,2) # (batch_size, n_heads, seq_len_v, d_k)


        attention_output = self.scaled_dot_product_attention(Q,K,V,mask) # 计算注意力输出 # (batch_size, n_heads, seq_len, d_k)

        # (batch_size, n_heads, seq_len, d_k) -> (batch_size, seq_len, d_model),把多个head的输出拼接，维度回到d_model
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)    # (batch_size, seq_len, d_model)
        output = self.W_o(attn_output)    # (batch_size, seq_len, d_model)
        return output    # (batch_size, seq_len, d_model)






    def scaled_dot_product_attention(self,Q,K,V,mask=None): #这里就是那个经典公式Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V
        # Q: (batch_size, n_heads, seq_len_q, d_k)
        # K: (batch_size, n_heads, seq_len_k, d_k)
        # V: (batch_size, n_heads, seq_len_v, d_k)

        # torch.matmul矩阵乘法
        scores = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.d_k) # 计算注意力分数,除以sqrt(d_k)进行缩放
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # apply mask to scores

        attn_weights = F.softmax(scores,dim=-1) #（batch_size, n_heads, seq_len_q, seq_len_k)
        attn_weights = self.dropout(attn_weights) # 应用dropout

        output = torch.matmul(attn_weights,V) #（batch_size, n_heads, seq_len_q, d_k)

        return output






