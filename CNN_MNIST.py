'''
2019-12-3

some simple models 
for MNIST classification.

Contact: jiazx@buaa.edu.cn
'''

import torch
import torchvision
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import numpy as np 
from torch.autograd import Variable

# A simple Fully Connected Model
class simple_net(nn.Module):
    def __init__(self, input_dim, h_1, h_2, output_dim):
        super(simple_net, self).__init__()
        self.layer1 = nn.Linear(input_dim, h_1)
        self.layer2 = nn.Linear(h_1, h_2)
        self.layer3 = nn.Linear(h_2, output_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x 

# FC net with some tricks.
class fc_net(nn.Module):
    def __init__(self, input_dim, h_1, h_2, output_dim):
        super(fc_net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, h_1),
            nn.BatchNorm1d(h_1),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(h_1, h_2),
            nn.BatchNorm1d(h_2),
            nn.ReLU(True)
        )
        # pytorch 在使用crossentropyloss损失函数时里面包含了softmax，最后一层不需要使用softmax
        self.layer3 = nn.Sequential(
            nn.Linear(h_2, output_dim)
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# CNN
class CNN_mnist(nn.Module):

    def __init__(self):
        super(CNN_mnist, self).__init__()
        # input (1,28,28)  out (8,14,14)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels= 1,
                out_channels= 8,
                kernel_size= (3, 3),
                stride= 1,
                padding= 1,
            ),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        # input (8,14,14)  out (16,7,7)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels= 8,
                out_channels= 16,
                kernel_size= (3, 3),
                stride= 1,
                padding= 1,
            ),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features= 16*7*7,
                out_features= 128,
            ),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(
                in_features=128,
                out_features=10,
            )
        )

    def forward(self,x):
        layers_out = []
        x = self.conv1(x)
        layers_out.append(x)
        x = self.conv2(x)
        layers_out.append(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.out(x)
        return x, layers_out


BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 1

data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5],[0.5])])

train_dataset = datasets.MNIST(root='./pytorch_exercise', train=True, transform=data_tf)
test_dataset = datasets.MNIST(root='./pytorch_exercise', train=False, transform=data_tf)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#加载模型
if torch.cuda.is_available():
    model = CNN_mnist().cuda() 
else:
    model = CNN_mnist()

# 定义损失函数 和 优化器
loss_func = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练模型
step = 0
for epoch in range(EPOCHS):

    for data in train_loader:
        img, label = data
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        else:
            img = Variable(img)
            label = Variable(label)

        # 数据在模型上前向传播
        out = model(img)[0]

        # 计算损失（目标函数）
        loss = loss_func(out, label)
        
        # 清空上一步优化器中的梯度
        opt.zero_grad()
        # 通过目标函数计算梯度
        loss.backward()
        # 优化器更新参数
        opt.step()

        step += 1
        if step%100 == 0:
            print('step: {}, loss: {:.4}'.format(step, loss.data.item()))

# 模型评估
model.eval()
eval_loss = 0
eval_acc = 0
for data in test_loader:
    img, label = data

    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()
    else:
        img = Variable(img)
        label = Variable(label)

    out = model(img)
    # 计算在一个batch size上的平均损失
    loss = loss_func(out, label)

    # 统计总的损失是多少，因此再乘上batch size
    eval_loss += loss.data.item()*label.size(0)

    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()

print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
    eval_loss / (len(test_dataset)),
    eval_acc / (len(test_dataset)))
    )

# conv1 and conv2 visualization
img = next(iter(test_loader))[0]
# 获取这两层的输出
conv1_out = model(img.cuda())[1][0]
conv2_out = model(img.cuda())[1][1]

import matplotlib.pyplot as plt
# 可视化输出
fig = plt.figure(figsize=(100, 100))
# fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
for i in range(16):
    ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
    ax.imshow(conv2_out[0][i].detach().cpu().numpy(), cmap="gray")

plt.show()
