import torchvision
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from model import *
#准备数据集
train_data = torchvision.datasets.CIFAR10(root="data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="data",train=False,transform=torchvision.transforms.ToTensor(),
                                         download=True)

#length长度
train_data_size = len(train_data)
test_data_size = len(test_data)
#如果train_data_size =10,训练集的长度为：10
print("训练集的长度为:{}".format(train_data_size))
print("测试集的长度为:{}".format(test_data_size))

#利用dataloader加载数据
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

#创建网络模型
tupu = Tupu()
tupu = tupu.cuda()

#损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn=loss_fn.cuda()
#优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(tupu.parameters(),lr=learning_rate)

#设置训练网络的一些参数
#记录训练次数
total_train_step = 0
#记录测试次数
total_test_step= 0
#训练次数
epoch = 10

#添加tensorboard
writer = SummaryWriter("logs_train")
for i in range(epoch):
    print("-----------第{}轮训练开始-----------".format(i+1))
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets=targets.cuda()

        outputs = tupu(imgs)
        loss = loss_fn(outputs,targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step+1
        if total_train_step % 100 ==0:
            print("训练次数:{},loss:{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)
    #测试开始
    total_test_loss =0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs=imgs.cuda()
            targets=targets.cuda()
            outputs = tupu(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss = total_test_loss+loss
            accuracy = (outputs.argmax(1)==targets).sum()
            total_accuracy+=accuracy
    print("整体测试集上的loss：{}".format(total_test_loss))
    print("正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
    total_test_step+=1

    torch.save(tupu,"tupu_pth".format(i))
    print("模型已保存")

writer.close()
