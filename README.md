# GAN
a repository for GAN  reproduction using pytorch

```bash
gan.py
GAN网络结构分为两部分,生成器网络Generator和判别器网络Discriminator.
-- 生成器Generator将随机生成的噪声z通过多个线性层生成图片,生成器的最后一层是Tanh,所以我们生成的图片的取值范围为[-1,1],
同理,我们会将真实图片归一化(normalize)到[-1,1].
-- 而判别器Discriminator是一个二分类器,通过多个线性层得到一个概率值来判别图片是"真实"或者是"生成"的,
所以在Discriminator的最后是一个sigmoid,来得到图片是"真实"的概率.


```