class Transformer(nn.Module):
    def __init__(self,src_vocab_size,tgt_vocab_size,d_model,n_heads,d_ff,num_layers,dropout=0.1):
        super().__init__()
        # 原始文本转token会由tokenizer完成，"I love cats"→[137, 521, 89]，然后将token送入embedding
        # token（输入句子分词后）的长度是不确定的，最后是由embeding把所有token拓展到d_model维特征，才能计算的
        self.encoder_embeding = nn.Embedding(src_vocab_size,d_model)
        self.decoder_embeding = nn.Embedding(tgt_vocab_size,d_model) # embedding将每个token多展开出d_model的特征，src(32,10)→(32,10,512)
        self.positional_encoding = self.Positional_Encoding(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.encoder = Encoder(d_model,n_heads,d_ff,num_layers,dropout)
        self.decoder = Decoder(d_model,n_heads,d_ff,num_layers,dropout)
        
        # decoder最后输出的是d_model维特征，但是我们想知道下一个token是词表中的哪一个，
        # 所以要做d_model到tgt_vocab_size的映射,这样就可以得到tgt_vocab_size个token分别对应的分数
        self.fc_out = nn.Linear(d_model,tgt_vocab_size)

    def forward(self,src,tgt,src_mask=None,tgt_mask=None):
        # src: (batch_size,src_seq_len)
        # tgt: (batch_size,tgt_seq_len)

        src = self.encoder_embedding(src) * math.sqrt(self.encoder_embedding.embedding_dim)
        # (batch_size,seq_len,d_model)
        # 为什么要×math.sqrt((self.encoder_embedding.embedding_dim))?
        # 本质上是Embedding(x)×根号下d_model，这是以为后续我们会有token Embedding + Position Encoding的操作
        # 位置编码的数值比较小，再[-1,1]之间，把token embedding放缩到更大对的尺度，让position编码和位置编码的尺寸相近
        # 避免梯度计算被位置编码主导
        tgt = self.decoder_embedding(tgt) * math.sqrt(self.decoder_eembedding.embedding_dim)

        src = self.dropout(src)   # (batch_size, src_seq_len, d_model)
        tgt = self.dropout(tgt)   # (batch_size, tgt_seq_len, d_model)

        src = self.positional_encoding(src)  # (batch_size, src_seq_len, d_model)
        tgt = self.positional_encoding(tgt)  # (batch_size, tgt_seq_len, d_model)

        enc_output = self.encoder(src,src_mask) # (batch_size, src_seq_len, d_model)
        dec_output = self.decoder(tgt,enc_output,tgt_mask) # (batch_size, tgt_seq_len, d_model)

        output = self.fc_out(dec_output)  # (batch_size, tgt_seq_len, tgt_vocab_size)
        return output  # (batch_size, tgt_seq_len, tgt_vocab_size)















