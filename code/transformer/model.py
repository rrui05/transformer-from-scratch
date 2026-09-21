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