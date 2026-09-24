# Reference:
https://huan-yin.github.io/2026/04/09/%E6%89%8B%E6%92%95DiT/

# diffusion transformer
基本思想： 用transformer替换掉传统diffusion中常用的U-Net架构
核心机制：
    * 1. 图像分patch，继承vit的思路
    * 2. adaLN-Zero：模型需要知道当前diffusion时间步t和条件y。adaLN可以理解为均值和方差的迁移
    * 3. Flow-Matching: 预测向量场 
            - 首先，x_0为初始的纯高斯噪声，x_1为truth，t~(0~1)
            - 那么中间步: x_t = (1-t)x_0 + t*x_1
            - 然后左右同时对t求导就可以得到diffusion速度场公式: v = x1-x0
            **flow matching dit 的训练目标就是 v_beta(x_t,t)=x1-x0**
            推理的时候从噪声一步步积分回去，就有x_0 -> x_1
            

 