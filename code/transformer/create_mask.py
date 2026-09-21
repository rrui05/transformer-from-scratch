def create_padding_mask(src,tgt,pad_idx=0):
    # src: (batch_size,src_seq_len)
    # tgt: (batch_size,tgt_seq_len)

    src_mask = (src !=pad_idx).unsqueeze(1).unsequeeze(2) # (batch_size,1,1,src_seq_len)
    tgt_mask = (tgt !=pad_idx).unsqueeze(1).unsequeeze(2) # (batch_size,1,tgt_seq_len,1)