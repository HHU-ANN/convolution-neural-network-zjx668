# 在该文件NeuralNetwork类中定义你的模型 
# 在自己电脑上训练好模型，保存参数，在这里读取模型参数（不要使用JIT读取），在main中返回读取了模型参数的模型


import os

os.system("sudo pip3 install torch")
os.system("sudo pip3 install torchvision")





import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
num_epochs = 25  # 50轮
batch_size = 50  # 50步长
learning_rate = 0.01  # 学习率0.01
from torch.utils.data import DataLoader
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def read_data():
    # 这里可自行修改数据预处理，batch大小也可自行调整
    # 保持本地训练的数据读取和这里一致
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor()])
    dataset_train = torchvision.datasets.CIFAR10(root='../data/exp03', train=True, download=True, transform=torchvision.transforms.ToTensor())
    dataset_val = torchvision.datasets.CIFAR10(root='../data/exp03', train=False, download=False, transform=torchvision.transforms.ToTensor())
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
    return dataset_train, dataset_val, data_loader_train, data_loader_val

# 3x3 卷积定义
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

   # Resnet 的残差块
class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1, downsample=None):
            super(ResidualBlock, self).__init__()
            self.conv1 = conv3x3(in_channels, out_channels, stride)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(out_channels, out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.downsample = downsample

        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)
            return out

class NeuralNetwork(nn.Module):
    # ResNet定义
    def __init__(self, block, layers, num_classes=10):
        super(NeuralNetwork, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

model = NeuralNetwork(ResidualBlock, [2, 2, 2]).to(device)

# 损失函数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 更新学习率
def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

train_dataset, test_dataset, train_loader, test_loader = read_data()
## 训练数据集
#total_step = len(train_loader)
#curr_lr = learning_rate
#for epoch in range(num_epochs):
#    for i, (images, labels) in enumerate(train_loader):
 #       images = images.to(device)
 #       labels = labels.to(device)
#
#     # Forward pass
  #      outputs = model(images)
  #      loss = criterion(outputs, labels)

        # Backward and optimize
  #      optimizer.zero_grad()
 #       loss.backward()
 #       optimizer.step()
#
  #      if (i + 1) % 100 == 0:
  #          print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
   #                 .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

   #     # 延迟学习率
  #  if (epoch + 1) % 20 == 0:
  #      curr_lr /= 3
 #       update_lr(optimizer, curr_lr)
#
#torch.save(model.state_dict(), 'model.pth')
#
#
#def read_data():
#    # 这里可自行修改数据预处理，batch大小也可自行调整
#    # 保持本地训练的数据读取和这里一致
 #   dataset_train = torchvision.datasets.CIFAR10(root='../data/exp03', train=True, download=True, transform=torchvision.transforms.ToTensor())
 #   dataset_val = torchvision.datasets.CIFAR10(root='../data/exp03', train=False, download=False, transform=torchvision.transforms.ToTensor())
 #   data_loader_train = DataLoader(dataset=dataset_train, batch_size=256, shuffle=True)
#    data_loader_val = DataLoader(dataset=dataset_val, batch_size=256, shuffle=False)
 #   return dataset_train, dataset_val, data_loader_train, data_loader_val

def main():
    model = NeuralNetwork(ResidualBlock, [2, 2, 2]).to(device) # 若有参数则传入参数
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model.load_state_dict(torch.load(parent_dir + '/pth/model.pth'))
    return model
    
