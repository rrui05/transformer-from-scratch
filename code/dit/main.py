import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import os

# 新建images 文件夹
os.makedirs("images", exist_ok=True)


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(42)


## ---正弦时间位置编码---- ##
# 将连续的(diffusion)时间步转化为高维向量，连时间变化都能建模
class SinusoidalPositionEmdeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        # 时间转向量：

        # 这里的操作其实和transformer位置编码的思路很像，都是转化到不同频率的sin，cos
        # 首先我们生成不同频率，那这里为什么是dim//2呢？
        # 因为每个频率同时对应了sin和cos，维度就乘2了
        # 先生成dim/2个不同频率，然后用时间t乘这些频率，放到sin，cos里
        # 最后得到token_t: [sin(tw_0),cos(t_w0),sin(tw_1),cos(tw_1)......sin(tw_m-1),cos(tw_m-1)]

        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)  # 生成频率base
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # 将时间t与base频率相乘
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        # mlp_ratio 是对中间层的放大倍数
        super().__init__()
        # 注意，普通的transformer中，LayerNorm有固定，可学习的gamma，beta（在LayerNorm的归一化公式中）
        # 而dit中是基于时间条件t动态生成scale/shift/gate，分别控制attention和MLP两个分支
        # 也就是adaLN
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

        # adaLN对6参数进行调制，输出维度为hidden_size*6
        # 对应：scale_mha,shift_mha,gate_mha,scale_mlp,shift_mlp,gate_mlp
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6)
        )

        # adaLN-Zero零初始化，保证训练初始的稳定性
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, t_emb):
        # 拆分六个调制参数
        modulation = self.adaLN_modulation(t_emb)
        scale_mha, shift_mha, gate_mha, scale_mlp, shift_mlp, gate_mlp = (
            modulation.chunk(6, dim=-1)
        )

        # 扩展维度，[:,None,:]是为了适配[B，1，hidden_size]，支持广播
        scale_mha = scale_mha[:, None, :]
        shift_mha = shift_mha[:, None, :]
        gate_mha = gate_mha[:, None, :]
        scale_mlp = scale_mlp[:, None, :]
        shift_mlp = shift_mlp[:, None, :]
        gate_mlp = gate_mlp[:, None, :]

        # self-attention 分支，使用专属3个参数
        residual = x
        x = self.norm1(x)
        x = x * (1 + scale_mha) + shift_mha  # adaLN调制
        attn_out, _ = self.attn(x, x, x)
        x = residual + gate_mha * attn_out  # 门控残差，决定用不用attention的结果

        # mlp分支
        residual = x
        x = self.norm2(x)
        x = x * (1 + scale_mlp) + shift_mlp
        mlp_out = self.mlp(x)
        x = residual + gate_mlp * mlp_out

        return x


class DiT(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        num_classes=4,  # 模拟条件生成
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 2

        # patch嵌入（利用步长等于核大小的卷积，不重叠）
        # 其实就是利用卷积去一次性完成切分+linear projection
        self.patcch_embed = nn.Conv2d(in_channels,hidden_size,kernel_size=patch_size,stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1,self.num_patches,hidden_size))

        # 时间+类别条件注入（输出给每个block的adaLN）
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmdeddings(hidden_size),
            nn.Linear(hidden_size,hidden_size)， # 前一步正余弦编码后，没有构造出可学习的参数，通过linear引入对时间token的操作和学习
            nn.GELU(),
            nn.Linear(hidden_size,hidden_size)
        )
        self.class_embed = nn.Embedding(num_classes,hidden_size)


        # Transformer Block
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size,num_heads,mlp_ratio) for _ in range(depth)
        ])

        # 输出头
        self.norm_final = nn.LayerNorm(hidden_size,elementwise_affine=False,eps=1e-6)
        # 最终输出层也适配adaLN，增加调制层
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size,hidden_size*2) # 最后一层不需要gate，所以只有scale和shift
        )

        # 输出维度为patch内的像素总数（p*p*C）
        self.head = nn.Linear(
            hidden_size,
            patch_size*patch_size*in_channels
        )

        ## 初始化权重
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed,std=0.02)
        nn.init.trunc_normal_(self.head.weight,std=0.02)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.final_modulation[-1].weight) # final_modulation[-1]表示final_modulation中的最后一个子层，也就是linear初始化为0
        nn.init.zeros_(self.final_modulation[-1].bias)

    def forward(self,x,t,y=None):
        B = x.shape[0]

        # 分片+位置编码
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1,2)
        x = x + self.pos_embed

        # 时间+条件嵌入（这两者是合并为统一的条件向量注入的）
        t_emb = self.time_embed(t)
        if y is not None:
            t_emb = t_emb +self.class_embed(y)

        # 经过Dit Blocks，每个block都接受统一的条件嵌入
        for block in self.blocks:
            x = block(x,t_emb)

        # 最终输出层的adaLN调制
        scale_final,shift_final = self.final_modulation(t_emb).chunk(2,dim=-1) #在最后一个维度切成两半
        x = self.norm_final(x)
        x = x * (1+scale_final[:,None,:])+shift_final[:,None,:]
        x = self.head(x)

        # 重组patch，还原回图像空间[B,C,H,W]
        h = w = int(self.num_patches ** 0.5)
        p = self.patch_size
        x = x.transpose(1,2).reshape(B,self.in_channels,p,p,h,w)
        x = torch.einsum('n c p q h w -> n c h p w q', x)
        x = x.reshape(B,self.in_channels,h*p,w*p)

        return x

def generate_simulation_dataset(
        num_samples = 2000,
        img_size = 32,
        in_channels = 4,
        num_classes = 4,
        noise_level = 0.05 # 添加少量噪声，模拟真实latent分布            
):
    """
    生成带类别规律的模拟Dit训练数据集
    """
    x = np.linspace(-np.pi,np.pi,img_size)
    y = np.linspace(-np.pi,np.pi,img_size)
    xx,yy = np.meshgrid(x,y)

    data,labels = [],[]
    samples_per_class = num_samples//num_classes

    for class_id in range(num_classes):
        for _ in range(samples_per_class):
            if class_id == 0:
                base_pattern = np.sin(2*xx)
            elif class_id == 1:
                base_pattern = np.sin(4*yy)
            elif class_id == 2:
                base_pattern = np.sign(np.sin(3 * xx))*np.sin(3 * yy)
            else:
                base_pattern = np.exp(-(xx**2 + yy**2) / 2) * np.cos(4 * np.sqrt(xx**2 + yy**2))

            multi_channel_pattern = np.stack([
                base_pattern + 0.1*i for i in range(in_channels)
            ],axis=0)

            multi_channel_pattern +=np.random.normal(0,noise_level,multi_channel_pattern.shape)
            multi_channel_pattern = multi_channel_pattern / np.max(np.abs(multi_channel_pattern))

            data.append(multi_channel_pattern)
            labels.append(class_id)

    data = torch.tensor(np.array(data),dtype = torch.float32)
    labels = torch.tensor(np.array(labels),dtype = torch.long)
    return data,labels

def flow_matching_loss(model,x_1,y=None):
    """
    Flow Matching 核心损失函数
    x_1：真实数据
    y：类别标签
    """
    B = x_1.shape[0]
    device = x_1.device

    # 1. 均匀采样时间步 t~U[0,1]
    t = torch.rand(B,device = device) # 这里的时间值是随机分配的，因为训练的时候不需要一步一步去噪
    t_expand = t[:,None,None,None] # 这一步是将维度从[B]变换到[B,1,1,1],目的是为了和后续的图像维度对齐（直接通过相乘广播），途中所有通道和像素共用同一个t

    # 2. 采样源噪声 x_0~N(0,1)
    x_0 = torch.randn_like(x_1)

    # 3. 构造线性插值路径x_t = x_0+t*(x_1-x_0)
    x_t = x_0 + t_expand*(x_1-x_0)

    # 4. 模型预测向量场
    v_pred = model(x_t, t, y )

    # 5. 损失：MSE(预测向量场，真实向量场x1-x0)
    v_true = x_1 - x_0
    loss = F.mse_loss(v_pred, v_true)

    return loss

#### 训练与推理采样
# 通过常微分方程（ODE）的欧拉方法（Euler Method）,根据预测的向量场逆向生成图像
# 数值计算方法，近似求解常微分方程
def train_dit(epochs = 50,batch_size = 64, lr = 1e-4, device = None):
    if device is None :
        device = torch.device("cuda",if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    print("正在生成模拟数据集")
    data,labels = generate_simulation_dataset(num_samples = 2000)
    dataset = TensorDataset(data,labels)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True,drop_last = True)

    model = DiT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr = lr,weight_decay = 1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
    grad_clip = 1.0 # 梯度剪裁

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params/1e6:.2f}M")

    print("开始训练...")
    model.train()
    loss_history=[]

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x1,batch_y in dataloader:
            batch_x1,batch_y = batch_x1.to(device),batch_y.to(device)

            optimizer.zero_grad()
            loss = flow_matching_loss(model,batch_x1,batch_y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
            optimizer.step()

            total_loss += loss.item() * batch_x1.shape[0]

        avg_loss = total_loss / len(dataset)
        loss_history.append(avg_loss)
        scheduler.step()

        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], 平均损失: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    print("训练完成！")
    return model,loss_history,data,labels

@torch.no_grad() # 装饰器的作用是，执行这个函数时，不记录用于反向传播的计算图
def dit_sample(model,num_samples = 8,class_id=0,img_size = 32,in_channels=4,num_steps=20,devie=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    # 1. 初始化，从标准高斯分布采样x_0(t=0)
    x = torch.randn(num_samples,in_channels,img_size,img_size,device = device)
    y = torch.tensor([class_id]*num_samples,device=device)

    # 3. 欧拉法时间步
    dt = 1.0 / num_steps
    t_list = torch.linspace(0,1 - dt,num_steps,device = device)

    # 4. 逐步积分求解ODE：dx/dt = v(x,t)
    for t in t_list:
        t_batch = torch.ones(num_samples,device=device)*t
        v_pred = model(x,t_batch,y)
        # 欧拉法更新
        x = x + v_pred * dt

    return x.cpu()

def visualize_all_classes(real_samples,gen_samples,num_classes=4):
    """
    对比可视化：所有类别的真实数据 vs 生成数据（放在同一张图）
    第一行：真实图像，第二行，生成图像
    """
    plt.figure(figsize = (num_classes*3,6))

    for cls_idx in range(num_classes):
        # 真实数据
        plt.subplot(2,num_classes,cls_idx + 1)
        plt.imshow(real_samples[cls_idx][0],cmap='viridis')
        plt.title(f'True Class {cls_idx}')
        plt.axis('off')

        # 生成数据
        plt.subplot(2,num_classes,cls_idx+1+num_classes)
        plt.imshow(gen_samples[cls_idx][0],cmap='viridis')
        plt.title(f'True Class {cls_idx}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("images/all_classes_comparison.png",dpi=150,bbox_inches='tight')
    plt.close()

# 损失曲线可视化
def plot_loss_curve(loss_history):
    plt.figure(figsize=(8,4))
    plt.plot(loss_history)
    plt.title("training loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Flow Matcching Loss")
    plt.grid(True,alpha = 0.3)
    plt.savefig("images/loss_curve.png",dpi=150,bbox_inches = 'tight')
    plt.close()

if __name__ == "__main__":
    # 1.训练模型
    trained_model,loss_history,real_data,real_labels = train_dit(epochs=50)

    # 2.绘制损失曲线，验证训练收敛性 
    plot_loss_curve(loss_history)

    # 3.为每个类别生成一个样本，并收集一个真实样本
    target_classes = [0,1,2,3]
    real_samples = []
    gen_samples = []

    print("\n 正在收集所有类别的样本...")

    for cls in target_classes:
        # 收集一个真实样本
        real_sample = real_data[real_labels == cls][0]
        real_samples.append(real_sample)

        # 生成一个样本
        gen_sample = dit_sample(trained_model,num_samples=1,class_id=cls)
        gen_samples.append(gen_sample[0])


    # 4.可视化所有类别在一张图里
    visualize_all_classes(real_samples,gen_samples)

    print("\n所有任务完成！生成的图片已保存到 images 文件夹")


















