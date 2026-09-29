![alt text](image.png)

U-Net 是一个对称的网络结构, 先下采样然后上采样, 在上采样的过程中, 通过skip connection操作融合下次采样中的
feature map.

skip connection: 就是将feature map的**通道进行叠加**, 俗称concat

大小不同也可以concat,方法就是大的剪裁或者小的padding(补0), U-Net中使用后者



