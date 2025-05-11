#from https://github.com/supercatex/ML_Examples/blob/master/demo/demo_sklearn_kmeans.py

# 匯入數值與資料處理的常用套件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 匯入 KMeans 聚類模型
from sklearn.cluster import KMeans

# 匯入經典的 iris 資料集（鳶尾花）
from sklearn.datasets import load_iris

# 匯入 PCA（主成分分析）降維工具
from sklearn.decomposition import PCA

# 匯入 t-SNE 非線性降維工具
from sklearn import manifold

# 載入 iris 資料集，這是一個常用的 3 類花種分類資料
data = load_iris()

# 將 iris 的前四個特徵轉為 pandas DataFrame，方便觀察與處理
df = pd.DataFrame(data["data"], columns=data["feature_names"])

# 將前四個特徵轉為 numpy 陣列 x（形狀為 [150, 4]）
x = np.array(df.iloc[:, 0:4])

# 將目標標籤（0, 1, 2）轉為 numpy 陣列 y（形狀為 [150, 1]）
y = np.array(data["target"]).reshape(-1, 1)

# ================================
# 使用 t-SNE 將資料降維為 3 維以便視覺化
# ================================

tsne = manifold.TSNE(n_components=3, init="pca")  # 建立 t-SNE 模型，輸出為 3 維，初始化方式為 PCA
x_tsne = tsne.fit_transform(x)                   # 對 x 進行降維，得到 x_tsne（形狀為 [150, 3]）

# 將 t-SNE 結果做最小-最大正規化到 [0, 1] 區間
x_min, x_max = x_tsne.min(), x_tsne.max()
x_norm = (x_tsne - x_min) / (x_max - x_min)

# 建立 3D 視覺化圖形
fig = plt.figure()
ax = fig.add_subplot(projection='3d')             # 建立 3D 子圖
ax.view_init(azim=45, elev=45)                    # 設定視角：方位角 45 度、高度角 45 度

# 對每一個資料點，用其對應的類別標籤（0/1/2）上色與標記文字
for i in range(x_norm.shape[0]):
    ax.text(
        x_norm[i, 0], x_norm[i, 1], x_norm[i, 2],  # 3D 座標
        str(y[i]),                                 # 顯示數字類別
        color=plt.cm.Set1(y[i]),                   # 用 Set1 調色盤上色（3 類）
        fontdict={"weight": "bold", "size": 9}     # 設定文字樣式
    )

plt.show()  # 顯示 t-SNE 的視覺化結果

# ================================
# 使用 PCA 將原始資料降維為 2 維
# ================================

pca = PCA(n_components=2)                # 建立 PCA 模型，將資料降至 2 維
new_x = pca.fit_transform(x)             # 對資料執行降維，並儲存在 new_x

# 輸出每個主成分解釋的變異量比例（越大代表該主成分越重要）
print(pca.explained_variance_ratio_)     # e.g. [0.92, 0.05] 表示第一個主成分解釋 92% 資訊

x = new_x                                # 後續的 K-Means 聚類將使用 PCA 降維後的 x

# ================================
# 使用 K-Means 對資料聚類，並計算每個 k 值對應的 inertia（誤差平方和）
# ================================

inertia = []               # 用來儲存不同 k 值的 inertia 值
n_classes = 10             # 設定最多測試到 k=10 群

# 嘗試 1~10 群的 KMeans 並儲存 inertia（用於找最佳 k）
for k in range(1, n_classes + 1, 1):
    kmeans = KMeans(n_clusters=k)   # 建立 KMeans 模型
    kmeans.fit(x)                   # 對降維後的資料進行聚類
    inertia.append(kmeans.inertia_)  # 記錄模型的 inertia 值（總內部距離）

# 繪製 Elbow Plot（手肘法），幫助選擇最適 k 值
plt.figure()
plt.plot(range(1, n_classes + 1, 1), inertia, marker='o')  # x 軸為 k 值，y 軸為 inertia
plt.show()

# ================================
# 繪製每個 k 值的聚類結果視覺化
# ================================

width = 5  # 每個圖的寬度
plt.figure(figsize=(width * (n_classes + 1) // 2, width * 2))  # 建立整體圖形大小
i = 0  # 子圖索引計數器

# 針對 k = 1 ~ 10 逐一建立子圖展示聚類結果
for k in range(1, n_classes + 1, 1):
    kmeans = KMeans(n_clusters=k)   # 建立 KMeans 模型
    kmeans.fit(x)                   # 對資料進行聚類

    i = i + 1
    ax = plt.subplot(2, (n_classes + 1) // 2, i)  # 建立子圖，分為 2 列
    ax.scatter(x[:, 0], x[:, 1], c=kmeans.labels_)  # 以聚類標籤上色每個點
