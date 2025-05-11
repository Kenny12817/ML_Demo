# Grok NOTES


# 機器學習（Scikit-learn）
## 1. 監督學習

### 線性回歸
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 載入資料集
diabetes = load_diabetes()
X = diabetes.data[:, np.newaxis, 2]  # 使用單一特徵
y = diabetes.target

# 分割資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立並訓練模型
model = LinearRegression()
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)
print("線性回歸預測結果：", y_pred[:5])
print("係數：", model.coef_)
print("均方誤差：", mean_squared_error(y_test, y_pred))
print("決定係數：", r2_score(y_test, y_pred))



### Logistic 回歸
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# 載入資料集
iris = load_iris()
X = iris.data
y = (iris.target == 0).astype(int)  # 二分類：setosa vs. 其他

# 分割資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立並訓練模型
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)
print("Logistic 回歸準確率：", accuracy_score(y_test, y_pred))



### KNN
from sklearn.neighbors import KNeighborsClassifier

# 載入資料集
X, y = iris.data, iris.target

# 分割資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立並訓練模型
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)
print("KNN 準確率：", accuracy_score(y_test, y_pred))



### 決策樹
from sklearn.tree import DecisionTreeClassifier

# 載入資料集
X, y = iris.data, iris.target

# 分割資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立並訓練模型
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)
print("決策樹準確率：", accuracy_score(y_test, y_pred))

### 隨機森林
from sklearn.ensemble import RandomForestClassifier

# 載入資料集
X, y = iris.data, iris.target

# 分割資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立並訓練模型
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)
print("隨機森林準確率：", accuracy_score(y_test, y_pred))




## 2. 非監督學習

### K-Means
from sklearn.cluster import KMeans

# 載入資料集
X = iris.data

# 建立並訓練模型
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# 獲取群集標籤
labels = kmeans.labels_
print("K-Means 群集標籤：", labels[:5])


### PCA
from sklearn.decomposition import PCA

# 載入資料集
X = iris.data

# 建立並訓練 PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("PCA 降維後形狀：", X_pca.shape)



### t-SNE
from sklearn.manifold import TSNE

# 載入資料集
X = iris.data

# 建立並訓練 t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
print("t-SNE 降維後形狀：", X_tsne.shape)



## 3. 模型估計

### 常用指標
from sklearn.metrics import classification_report

# 訓練隨機森林模型
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 評估模型
report = classification_report(y_test, y_pred)
print("分類報告：\n", report)



### 交叉驗證
from sklearn.model_selection import cross_val_score

# 建立模型
model = RandomForestClassifier()

# 交叉驗證
scores = cross_val_score(model, X, y, cv=5)
print("交叉驗證分數：", scores)



### 超參數調整
from sklearn.model_selection import GridSearchCV

# 定義超參數網格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20]
}

# 建立模型
model = RandomForestClassifier()

# 超參數調整
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X, y)
print("最佳參數：", grid_search.best_params_)



# 深度學習（PyTorch）

## 1. 神經網絡

### 簡單神經網絡
import torch
import torch.nn as nn
import torch.optim as optim

# 定義模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

# 初始化模型
model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 假設資料
X = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[2.0], [4.0], [6.0]])

# 訓練
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')


### 激活函數
import torch.nn.functional as F

x = torch.tensor([-1.0, 0.0, 1.0])
print("ReLU:", F.relu(x))
print("Sigmoid:", torch.sigmoid(x))
print("Tanh:", torch.tanh(x))


### 損失函數
# MSELoss
criterion = nn.MSELoss()
output = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([1.1, 2.1, 3.1])
loss = criterion(output, target)
print("MSELoss:", loss.item())

# CrossEntropyLoss
criterion = nn.CrossEntropyLoss()
output = torch.randn(3, 5)
target = torch.tensor([1, 0, 4])
loss = criterion(output, target)
print("CrossEntropyLoss:", loss.item())



## 2. 深度學習

### 多層感知機（MLP）
from torchvision import datasets, transforms

# 定義模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 載入資料集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 初始化模型
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 訓練
for epoch in range(5):
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/5], Loss: {loss.item():.4f}')

### 優化器
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01)
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)



## 3. 計算機視覺

### CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 12 * 12)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)



### 遷移學習（ResNet）
import torchvision.models as models

# 載入預訓練模型
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# 凍結特徵提取層
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


### 圖像增廣
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



## 4. 自然語言處理

### 詞嵌入（word2vec）
from gensim.models import Word2Vec

sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
model = Word2Vec(sentences, min_count=1)
print(model.wv.most_similar("cat"))

### Transformer（簡單實現）
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        Q = self.W_q(Q).view(-1, self.num_heads, self.d_k)
        K = self.W_k(K).view(-1, self.num_heads, self.d_k)
        V = self.W_v(V).view(-1, self.num_heads, self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        output = output.view(-1, self.num_heads * self.d_k)
        output = self.W_o(output)
        return output



### 預訓練模型（BERT）
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
print(logits)
