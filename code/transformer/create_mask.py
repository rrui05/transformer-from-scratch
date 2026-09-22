import torch

def create_padding_mask(src,tgt,pad_idx=0):
    # src: (batch_size,src_seq_len)
    # tgt: (batch_size,tgt_seq_len)

    # 注意基本流程，mask 是在QK^T得到 attention score 矩阵之后，再去把某些位置盖掉。
    # 盖掉之后再进入softmax！！！！！！

    # 这里的是先pad mask，PAD是用来把src和tgt调整为一样长的占位token
    src_mask = (src !=pad_idx).unsqueeze(1).unsqueeze(2) # (batch_size,1,1,src_seq_len)
    tgt_mask = (tgt !=pad_idx).unsqueeze(1).unsqueeze(2) # (batch_size,1,tgt_seq_len,1)
    
    # 为什么是(batch_size,1,1,src_seq_len)和(batch_size,1,tgt_seq_len,1)这两个大小呢？
    # 首先理解attention matrix：
    # 它展示的只是用横坐标Q查询纵坐标K得到的结果矩阵，细节见transformer\MHA.py:67
    # 需要注意的是，pytorch矩阵乘法matmul函数在计算的时候，只把最后两维当行和列计算
    # attention(Q, K, V)
    # Q 决定 attention matrix 的行数，竖着放作为列
    # K 决定 attention matrix 的列数，横着放作为行
    
    # 我们可以记住attention score的形状(batch_size, n_heads, query_len, key_len)
    # 对照便知，第2个维度的'1'会被广播至n_heads,所有head只是拿到的那不同的特征，也就是QKV不同的子空间，通过划分d_model得到
    # 而所有head用的都是同样的mask
    # 所有做mask的时候不用区分head，都填1，后续直接广播
    # 最后两个维度query_len, key_len 就是1,src_seq_len和tgt_seq_len,1
    # 这里默认src是Q，tgt是K
    # 所以src要遮住几整列（pad），tgt要遮住整几行（pad）
    
    
    # causal attention mask的作用是屏蔽掉src作为key的位置
    tgt_len = tgt.size(1)
    look_ahead_mask = torch.ones(tgt_len, tgt_len).tril().bool().unsqueeze(0).unsqueeze(0)    # (1, 1, tgt_len, tgt_len)先得到下三角阵
    tgt_mask = tgt_mask & look_ahead_mask.to(tgt.device)    # (batch_size, 1, tgt_len, tgt_len)
    
    return src_mask, tgt_mask
    
    
    
    
    
    
    
    
    
    
    
    