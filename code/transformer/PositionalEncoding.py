class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(
            0,
            max_len,
            dtype=torch.float
        ).unsqueeze(1)  # 最终得到shape(max_len,1),torch.arange(0,max_len)生成一个从0到max_len-1的整数序列，unsqueeze(1)将其变为列向量(增加一个维度)
        div_term = torch.exp(
            torch.arange(0,d_model,2).float() * -(math.log(10000.0)/d_model)
        )  # 对应原论文的位置编码公式

        pe = torch.zeros(1,max_len,d_model) # 创建位置编码矩阵，shape为(1,max_len,d_model)
        pe[0,:,0::2] = torch.sin(position*div_term) # 偶数位置使用sin函数
        pe[0,:,1::2] = torch.cos(position*div_term) # 奇数位置使用cos函数
        self.register_buffer('pe', pe) # 不参与梯度计算
        # d_model: embedding维度,用512个不同频率的sin/cos值共同表示同一个位置
    def forward(self, x):
        x = x + self.pe[:,:x.size(1),:] # 将位置编码加到输入的embedding上，x.size(1)表示序列长度
        return x