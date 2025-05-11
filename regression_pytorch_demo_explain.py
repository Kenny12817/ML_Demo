#from https://github.com/supercatex/ML_Examples/blob/master/demo/demo_pytorch_regression.py

# 載入必要的套件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# 自定義資料集類別
class MyDataset(Dataset):
    def __init__(self, csv_path):
        # 讀取 CSV 檔案
        self.df = pd.read_csv(csv_path)
        
        # 取第2欄作為輸入 x、第3欄作為輸出 y（假設第一欄是索引或非特徵）
        self.x = self.df.iloc[:, 1].values.reshape(-1, 1)
        self.y = self.df.iloc[:, 2].values.reshape(-1, 1)

        # 對 x 資料做標準化處理 (平均為 0，標準差為 1)
        self.x_mean, self.x_std = self.x.mean(), self.x.std()
        self.x_min, self.x_max = self.x.min(), self.x.max()
        self.x = (self.x - self.x_mean) / self.x_std
        # 若要使用 Min-Max 正規化，請取消下面這行註解
        # self.x = (self.x - self.x_min) / (self.x_max - self.x_min)

        # 同理對 y 資料做標準化處理
        self.y_mean, self.y_std = self.y.mean(), self.y.std()
        self.y_min, self.y_max = self.y.min(), self.y.max()
        self.y = (self.y - self.y_mean) / self.y_std
        # self.y = (self.y - self.y_min) / (self.y_max - self.y_min)

        # 將 x 和 y 合併成一個列表，並依據 x 值做排序
        self.data = list(np.concatenate((self.x, self.y), axis=1))
        self.data.sort(key=lambda x: x[0])
        self.data = np.array(self.data)

        # 將 x 和 y 轉換成 PyTorch tensor
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    # 回傳資料長度
    def __len__(self):
        return self.df.shape[0]

    # 取得特定索引的資料
    def __getitem__(self, index):
        return self.x[index], self.y[index]

# 定義模型架構（多項式線性回歸）
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.k = 5  # 多項式階數為 5（包含一次項到五次項）
        self.linear_layer = nn.Linear(self.k, 1)  # 輸入 k 維，輸出 1 維

    def forward(self, x):
        xx = x
        # 依序加入 x 的 2 次方、3 次方...直到 k 次方
        for i in range(2, self.k + 1, 1):
            xx = torch.cat((xx, x ** i), dim=1)
        x = xx
        # 線性層輸出
        x = self.linear_layer(x)
        return x

# 建立資料集物件
dataset = MyDataset("linear_regression_dataset_sample.csv")

# 切分為訓練資料與驗證資料（80% / 20%）
n = int(len(dataset) * 0.8)
train_data, valid_data = random_split(dataset, [n, len(dataset) - n])

# 使用 DataLoader 處理批次訓練
train_loader = DataLoader(train_data.dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data.dataset, batch_size=32, shuffle=True)

# 建立模型、損失函數與優化器
model = MyModel()
criterion = nn.MSELoss()  # 使用均方誤差
optimizer = optim.Adam(model.parameters(), lr=0.01)  # 使用 Adam 優化器
num_epochs = 1000  # 訓練輪數

train_loss_list = []
valid_loss_list = []

# 開始訓練迴圈
for epoch in range(num_epochs):
    model.train()  # 設定為訓練模式
    total_train_loss = 0
    for i, (X, y) in enumerate(train_loader):
        h = model(X)  # 預測
        train_loss = criterion(h, y)  # 計算損失
        optimizer.zero_grad()  # 清除梯度
        train_loss.backward()  # 反向傳播
        optimizer.step()  # 更新權重
        total_train_loss += train_loss.item()
    train_loss_list.append(total_train_loss)

    model.eval()  # 設定為評估模式
    total_valid_loss = 0
    for i, (X, y) in enumerate(valid_loader):
        h = model(X)  # 預測
        valid_loss = criterion(h, y)  # 計算損失
        total_valid_loss += valid_loss.item()
    valid_loss_list.append(total_valid_loss)

    # 每 100 輪印出一次損失值
    if epoch % 100 == 0:
        print("Epoch %5d => train_loss: %.4f, valid_loss: %.4f" % (
            epoch, total_train_loss,  total_valid_loss
        ))

# 畫出訓練與驗證的損失變化圖
plt.plot(train_loss_list, c="b")  # 訓練損失（藍色）
plt.plot(valid_loss_list, c="r", alpha=0.5)  # 驗證損失（紅色）

# 畫出原始資料與模型預測的對比圖
plt.figure()
data = dataset.data
plt.scatter(data[:, 0], data[:, 1])  # 畫出資料點
x = torch.tensor(data[:, 0:-1], dtype=torch.float32)
h = model(x).detach().numpy()  # 模型預測值
plt.plot(data[:, 0], h)  # 畫出模型預測曲線
