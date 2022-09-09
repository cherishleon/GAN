import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
import os

# 创建文件夹
if not os.path.exists('./img'):
    os.mkdir('./img')

def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)  # Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内：
    out = out.view(-1, 1, 28, 28)  
    return out

#config
batch_size = 128
num_epoch = 200
dimensions = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

img_transform = transforms.Compose([
    transforms.ToTensor(),
    # normalize with mean=0.5 std=0.5
    transforms.Normalize((0.5,), (0.5,))  # (x-mean) / std 
])


#mnist数据集下载,若需下载只需将download设为True
mnist = datasets.MNIST(
    root='.\dataset\mnist', train=True, transform=img_transform, download=False)

dataloader = torch.utils.data.DataLoader(
    dataset=mnist, batch_size=batch_size, shuffle=True)


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784, 512),  # 输入特征数为784，输出为256
            nn.LeakyReLU(0.2), 
            nn.Linear(512, 256),  
            nn.LeakyReLU(0.2),  
            nn.Linear(256, 1),
            nn.Sigmoid()  #得到一个0到1之间的概率进行二分类
        )

    def forward(self, x):
        x = self.dis(x)
        return x




# 然后经过线性变换将其变成784维，最后经过Tanh激活函数是希望生成的假的图片数据分布
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 256),  
            nn.BatchNorm1d(256),
            nn.ReLU(True),  
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),  
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),  
            nn.Linear(512, 784), 
            nn.Tanh()  # Tanh激活使得生成数据分布在[-1,1]之间，因为输入的真实数据的经过transforms之后也是这个分布
        )

    def forward(self, x):
        x = self.gen(x)
        return x


# 创建对象
D = discriminator()
G = generator()
if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()


criterion = nn.BCELoss()  # 二分类交叉熵函数
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)


for epoch in range(num_epoch):  
    for i, (img, _) in enumerate(dataloader):
        num_img = img.size(0)

        # view()函数作用是将一个多行的Tensor,拼接成一行
        img = img.view(num_img, -1)  # 将图片展开为28*28=784
        real_img = img.to(device)  #将tensor放入计算图中
        real_label = torch.ones(num_img).to(device) 
        fake_label = torch.zeros(num_img).to(device) 

        # ########判别器训练##############
        # 计算真实图片的损失
        real_out = D(real_img)  
        real_out = real_out.squeeze()  # (128,1) -> (128,)
        d_loss_real = criterion(real_out, real_label)  
        real_scores = real_out  # 得到真实图片的判别值，输出的值越接近1越好

        # 计算假的图片的损失
        noise = torch.randn(num_img, dimensions).to(device) # 随机生成一些噪声一个100维的高斯分布数据
        fake_img = G(noise).detach()  # 生成一张假的图片。因为G不用更新, detach分离
        fake_out = D(fake_img)  
        fake_out = fake_out.squeeze()  
        d_loss_fake = criterion(fake_out, fake_label) 
        fake_scores = fake_out  # 得到假图片的判别值，对于判别器来说，假图片的损失越接近0越好


        d_loss = d_loss_real + d_loss_fake  # 损失包括判真损失和判假损失
        d_optimizer.zero_grad()  #在反向传播之前，先将梯度归0
        d_loss.backward()  # 将误差反向传播
        d_optimizer.step()  # 更新参数



        # 在此过程中，将判别器固定，将假的图片传入判别器的结果与真实的label对应，
        # 反向传播更新的参数是生成网络里面的参数，
        # 这样可以通过更新生成网络里面的参数，来训练网络，使得生成的图片让判别器以为是真的
        noise = torch.randn(num_img, dimensions).to(device)
        fake_img = G(noise)  # 随机噪声输入到生成器中，得到一副假的图片
        output = D(fake_img)  # 经过判别器得到的结果
        output = output.squeeze()
        g_loss = criterion(output, real_label)  # 得到的假的图片与真实的图片的label的loss

        g_optimizer.zero_grad()
        g_loss.backward() 
        g_optimizer.step()  

        # 打印中间的损失
        if (i + 1) % 100 == 0:
            print('Epoch[{}/{}],d_loss:{:.4f},g_loss:{:.6f} '
                    'D real: {:.4f},D fake: {:.4f}'.format(
                epoch, num_epoch, d_loss.data.item(), g_loss.data.item(),
                real_scores.data.mean(), fake_scores.data.mean()  # 打印的是真实图片的损失均值
            ))
        if epoch == 0:
            real_images = to_img(real_img.cpu().data)
            save_image(real_images, './img/real_images.png')
    fake_images = to_img(fake_img.cpu().data)
    save_image(fake_images, './img/fake_images-{}.png'.format(epoch + 1))

# 保存模型
torch.save(G.state_dict(), './generator.pth')
torch.save(D.state_dict(), './discriminator.pth')