import torch
from torchvision.datasets import MNIST  # torchvision提供了许多vision数据集接口
from torchvision import transforms
from torch.utils.data import DataLoader  # 别调用成小写的dataloader了
from torch.utils.data import Dataset
from torch import nn  # 神经网络
import os
from PIL import Image

"""
深度学习是由数据支撑起来的，做深度学习往往伴随着大量的数据。如果把所有的数据全加载到内存上，容易把内存撑爆，所以要分批次一点点加载数据。
而且每种深度学习的框架都有自己所规定的数据格式，数据加载器DataLoader需要把大量的数据，分批次加载和处理成框架所需要的数据格式。

DataLoader不是只装一个批次的数据多次来运送，而是有处理好格式的分了批次的所有数据
通过for data in DataLoader，每次迭代返回的data是一个批次的数据

模型的构建、训练的流程：1.加载数据 2.数据预处理 3.实例化模型、优化器、损失函数、
4.前向传播得预测值、计算损失loss、5、反向传播loss.backward、更新参数6、模型的保存、模型的加载、模型的评估(不需要进行梯度的计算)
"""

# 图片预处理
transform = transforms.Compose([  # compose把多个transform操作合并
    transforms.ToTensor(),  # 转为tensor张量
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化归一化
])

# 训练数据和测试数据
train_data = MNIST(root='./data', train=True, download=False, transform=transform)  # 下载数据并进行transform操作
train_loader = DataLoader(train_data, shuffle=True, batch_size=64)  # 构建数据加载器，打乱数据，训练单次传递的样本个数为64
test_data = MNIST(root='./data', train=False, download=False, transform=transform)
test_loader = DataLoader(test_data, shuffle=False, batch_size=64)


"""
网络的固定格式： 
class Model(nn.Module):     
    def __init__(self):
        super(Model, self).__init__()   父类的初始化
    
    def forward(self, x):               前向传播就是：input -> forward -> output
"""


# 神经网络模型
# 图片是28*28,784个维度因此输入是784，最终输出是10(10维向量)，三层神经网络从784到10个维度。
class Model(nn.Module):
    def __init__(self):  # 在__init__函数中，通过self.变量名，设置的变量是全局变量，方便下面forward函数使用。
        super(Model, self).__init__()
        self.linear1 = nn.Linear(784, 256)  # 三个线性层
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 10)  # 10个手写数字对应10个输出.

    def forward(self, x):  # 前向传播
        x = x.view(-1, 784)  # 将输入数据拉平，列数784列
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        return x


model = Model()  # 模型实例化
criterion = nn.CrossEntropyLoss()  # 分类用交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(), 0.1)  # 优化器，随机梯度下降，学习率为0.2




def train(epoch):
    for i in range(1, epoch + 1):
        # enumerate作用是将一个可遍历的数据对象变为（索引，数据）的格式。
        for index, data in enumerate(train_loader):
            input, target = data
            optimizer.zero_grad()  # 训练阶段：优化器清空梯度
            y_predict = model(input)  # 前向传播，计算预测值
            loss = criterion(y_predict, target)  # 利用预测值和真实值计算损失
            loss.backward()  # 反向传播，回传损失，
            optimizer.step()  # 梯度下降，优化器更新参数。
            if index % 100 == 0:
                torch.save(model.state_dict(), "model/model_mlp.pth")  # 保存模型的参数字典
                print(f"loss:", loss.item())  # 打印损失


def test():
    correct = 0  # 正确的个数
    total = 0  # 总数
    with torch.no_grad():  # 测试不用计算梯度
        for data in test_loader:
            input, target = data
            output = model(input)  # output输出10个预测取值，其中最大概率值的索引即为预测的数
            _, predict = torch.max(output.data, dim=1)  # dim=1，按列找最大值
            total += target.size(0)  # 求总数
            correct += (predict == target).sum().item()  # 判断predict和targe相等的个数。
    print("准确率：", correct / total)


"""
if __name__ =='__main__'的意思是：
直接右键运行.py文件的时候，if __name__ =='__main__'：下方的代码将被执行
但当.py文件以模块形式被导入时，if __name__ =='__main__'下代码不被执行
比如main函数调用自定义数据集类，自定义数据集类下面的if __name__ =='__main__'就不会执行
通过这个特性，if __name__ =='__main__'就可以用于调试其他被调用的类写的是否正确，而不会在主函数运行。
"""

if __name__ == '__main__':
    # 如果模型存在，就加载模型，否则训练模型
    if os.path.exists('model/model_mlp.pth'):
        model.load_state_dict(torch.load("model/model_mlp.pth"))  # 加载模型
    else:
        train(10)
        test()

    # 测试单张图像
    img = Image.open("rewrite_number.png").convert("L")
    img = transform(img)  # 预处理
    img.view(-1, 784)  # 变形784维
    result = model(img)  # 调用模型进行预测
    a, predict = torch.max(result.data, dim=1)  # 从输出结果里拆包
    print(result)  # 输出各个位置的概率
    print(a)  # 输出最大的概率
    print("the result is:", predict.item())  # 打印索引，其实就是输出他判断的数字


