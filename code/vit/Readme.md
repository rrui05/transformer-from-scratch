## Reference
* https://zhuanlan.zhihu.com/p/637787022


## note
Although vit is based on transformer，but their transformer blocks are still different.
- Transfoemer blocks in vit use Pre-Norm(LayerNorm → attention/MLP) while transformer chose Pose-Norm(attention/MLP → LayerNorm).
- Pre-Norm don't have to warmup while Pose-Norm have to do that,beside warmup Pose-Norm needs other skill tin training.It's harder to train but has stronger generalization ability.




## run






