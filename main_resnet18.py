import torch
import torchvision.models
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, ReLU
from torch.nn import MaxPool2d
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
import os
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 图片预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 训练数据和测试数据
train_data = MNIST(root='./data', train=True, download=False, transform=transform)
train_loader = DataLoader(train_data, shuffle=True, batch_size=64)
test_data = MNIST(root='./data', train=False, download=False, transform=transform)
test_loader = DataLoader(test_data, shuffle=False, batch_size=64)


# 神经网络模型
model = torchvision.models.resnet18(pretrained=False)
# MNIST是灰度图，而resnet是3通道的，所以要修改第一层卷积层
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# 获得模型的全连接层输出特征数
num_ftrs = model.fc.in_features
# 修改为全连接层为：(num_ftrs, 10)
model.fc = torch.nn.Linear(num_ftrs, 10)
model = model.to(device)

# 损失函数和优化器实例化
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
optimizer = torch.optim.SGD(model.parameters(), 0.01)


def train(epoch):
    for i in range(1, epoch + 1):
        for index, data in enumerate(train_loader):
            input, target = data
            input = input.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            y_predict = model(input)
            loss = criterion(y_predict, target)
            loss.backward()
            optimizer.step()
            if index % 100 == 0:
                torch.save(model.state_dict(), "./model/model_resnet18.pth")
                print("loss:", loss.item())


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            input, target = data
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            _, predict = torch.max(output.data, dim=1)
            total += target.size(0)
            correct += (predict == target).sum().item()
    print("准确率：", correct / total)


if __name__ == '__main__':
    # 如果模型存在，就加载模型，否则训练模型
    if os.path.exists('model/model_resnet18.pth'):
        model.load_state_dict(torch.load("model/model_resnet18.pth"))  # 加载模型
    else:
        train(10)
        test()

    # 测试单张图像
    img = Image.open("rewrite_number.png").convert("L")
    img = transform(img)
    img.view(-1, 784)
    result = model(img)
    a, predict = torch.max(result.data, dim=1)
    print(result)
    print(a)
    print("the result is:", predict.item())


