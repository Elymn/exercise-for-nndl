import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签SimHei
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

# 设备配置（使用GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义目标函数
def targetFunc(x):
    """
    目标函数：f(x) = x^3
    """
    return x ** 3

# 生成数据点（从-10到10，步长为0.1）
x = np.arange(0, 5 * np.pi, 0.001)
y = [targetFunc(i) for i in x]

# 划分数据集：训练集（80%）、验证集（10%）、测试集（10%）
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=1)

print(f"训练集大小: {X_train.shape}")
print(f"验证集大小: {X_val.shape}")
print(f"测试集大小: {X_test.shape}")

# 自定义数据集类
class CurveDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.tensor(X, dtype=torch.float32)  # 输入数据
        self.y = torch.tensor(y, dtype=torch.float32)  # 目标值

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]  # 返回第idx个样本

    def __len__(self):
        return len(self.y)  # 返回数据集大小

# 创建数据集
train_dataset = CurveDataset(X_train, y_train)
val_dataset = CurveDataset(X_val, y_val)
test_dataset = CurveDataset(X_test, y_test)

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 训练集数据加载器
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)  # 验证集数据加载器
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # 测试集数据加载器

# 定义两层ReLU神经网络
class MLP(nn.Module):
    """
    两层全连接神经网络：
    - 输入层：1个神经元（x）
    - 隐藏层：10个神经元，使用ReLU激活函数
    - 输出层：1个神经元（预测的y）
    """
    def __init__(self, in_features=1, out_features=1):
        super().__init__()
        self.FC1 = nn.Linear(in_features=in_features, out_features=10)  # 第一层全连接
        self.relu = nn.ReLU()  # ReLU激活函数
        self.FC2 = nn.Linear(in_features=10, out_features=out_features)  # 第二层全连接

    def forward(self, x):
        x = self.FC1(x)  # 通过第一层
        x = self.relu(x)  # 应用ReLU激活
        outputs = self.FC2(x)  # 通过第二层
        return outputs

# 初始化模型
model = MLP().to(device)
print(model)

# 权重初始化函数
def weights_init(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight.data, 0.0, 0.02)  # 使用正态分布初始化权重

model.apply(weights_init)

# 定义损失函数（L1损失）
loss_fn = nn.L1Loss()

# 验证函数
def val(model, dataloader, loss_fn, device='cpu'):
    """
    验证模型性能
    - model: 神经网络模型
    - dataloader: 验证集数据加载器
    - loss_fn: 损失函数
    - device: 设备（CPU或GPU）
    """
    model.to(device)
    model.eval()
    with torch.inference_mode():
        rec_loss = 0
        for X, y in dataloader:
            X = X.to(device).unsqueeze(-1).float()  # 调整输入形状
            y = y.to(device).unsqueeze(-1).float()  # 调整目标形状
            logits = model(X)  # 模型预测
            loss = loss_fn(logits, y)  # 计算损失
            rec_loss += loss
        val_loss = rec_loss / len(dataloader)  # 平均损失
        print(f"验证集损失: {val_loss}")
        return val_loss

# 训练函数
def training(model, dataloader, val_dataloader, loss_fn, lr=0.001, epochs=50, device='cpu', verbose_epoch=10):
    """
    训练模型
    - model: 神经网络模型
    - dataloader: 训练集数据加载器
    - val_dataloader: 验证集数据加载器
    - loss_fn: 损失函数
    - lr: 学习率
    - epochs: 训练轮数
    - device: 设备（CPU或GPU）
    - verbose_epoch: 每隔verbose_epoch轮打印一次损失
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 使用Adam优化器
    best_val_loss = float('inf')  # 记录最佳验证损失

    for epoch in tqdm(range(epochs)):
        model.train()
        rec_loss = 0
        for X, y in dataloader:
            X = X.to(device).unsqueeze(-1).float()  # 调整输入形状
            y = y.to(device).unsqueeze(-1).float()  # 调整目标形状

            logits = model(X)  # 模型预测
            loss = loss_fn(logits, y)  # 计算损失
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            rec_loss += loss

        if epoch % verbose_epoch == 0:
            print(f"Epoch{epoch}\tLoss{rec_loss / len(dataloader)}")
            val(model, val_dataloader, loss_fn, device)

# 测试函数
def test(model, ranger, steper, loss_fn, device='cpu'):
    """
    测试模型性能
    - model: 神经网络模型
    - ranger: 测试范围（如 [-10, 10]）
    - steper: 步长（如 0.1）
    - loss_fn: 损失函数
    - device: 设备（CPU或GPU）
    """
    model.to(device)
    model.eval()
    x = []
    y_pred = []
    with torch.inference_mode():
        rec_loss = 0
        for X in np.arange(ranger[0], ranger[1], steper):
            X = torch.tensor(X).to(device).unsqueeze(-1).float()  # 调整输入形状
            y = torch.tensor([targetFunc(i) for i in X]).to(device).unsqueeze(-1).float()  # 目标值

            logits = model(X)  # 模型预测
            loss = loss_fn(logits, y)  # 计算损失
            rec_loss += loss

            x.extend(X.unsqueeze(1))
            y_pred.extend(logits.unsqueeze(1))

        print(f"测试集损失: {rec_loss * steper / (ranger[1] - ranger[0])}")

    x = [i.cpu().numpy() for i in x]
    y_pred = [i.cpu().numpy() for i in y_pred]

    # 绘制真实值和预测值对比图
    plt.plot(x, [targetFunc(i) for i in x], label="真实值")
    plt.plot(x, y_pred, label="预测值")
    plt.legend()
    plt.show()

# 训练模型
training(model, train_dataloader, val_dataloader, loss_fn, lr=0.001, epochs=100, device=device, verbose_epoch=10)

# 测试模型
test(model, [0, 10], 1, loss_fn, device)