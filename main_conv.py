import torch
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, ReLU
from torch.nn import MaxPool2d
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
import os
from PIL import Image

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

'''


class CONV(nn.Module):
    def __init__(self):
        super(CONV, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1, 1)  （输入图像的通道数，输出的通道数，卷积核尺寸，步长，填充）  
        self.conv2 = nn.Conv2d(1, 6, 3, 1, 1)   卷积核中的值是根据图像采样自动得到的，无需自己设置，在训练过程中还会自动调整
                                                通道数比原来变多，是因为同时用多个不同卷积核去卷积
    def forward(self, x):                       (3,1,1)和(5,1,2)时，是特征图尺寸不发生变化的常用配置
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))

conv = CONV()
output = conv(img)
'''


# 神经网络模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model1 = Sequential(
            Conv2d(1, 16, 3, 1, 1),
            ReLU(),
            MaxPool2d(3,1,1),

            Conv2d(16, 64, 3, 1, 1),
            ReLU(),
            MaxPool2d(3,1,1),

            Conv2d(64, 16, 3, 1, 1),
            ReLU(),
            MaxPool2d(3,1,1),

            Flatten(),
            Linear(784, 10),
        )

    def forward(self, x):
        x = self.model1(x)
        return x


model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), 0.01)

def train(epoch):
    for i in range(1, epoch + 1):
        for index, data in enumerate(train_loader):
            input, target = data
            optimizer.zero_grad()
            y_predict = model(input)
            loss = criterion(y_predict, target)
            loss.backward()
            optimizer.step()
            if index % 100 == 0:
                torch.save(model.state_dict(), "./model/model_conv.pth")
                print("loss:", loss.item())


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            input, target = data
            output = model(input)
            _, predict = torch.max(output.data, dim=1)
            total += target.size(0)
            correct += (predict == target).sum().item()
    print("准确率：", correct / total)


if __name__ == '__main__':
    # 如果模型存在，就加载模型，否则训练模型
    if os.path.exists('model/model_conv.pth'):
        model.load_state_dict(torch.load("model/model_conv.pth"))  # 加载模型
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

'''
N：表示批量大小（Batch Size），即输入数据中样本的数量。
C：表示通道数（Channels），即输入或输出数据中的通道数量。
H：表示高度（Height），即输入或输出数据的高度维度。
W：表示宽度（Width），即输入或输出数据的宽度维度。
'''