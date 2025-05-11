#from https://github.com/supercatex/ML_Examples/blob/master/demo/demo_sklearn_classifier.py

# 匯入 numpy 用於數值計算
import numpy as np

# 匯入 matplotlib 用於繪圖
import matplotlib.pyplot as plt

# 匯入 ListedColormap 用於自訂分類顏色
from matplotlib.colors import ListedColormap

# 匯入三種可產生非線性分類資料的函式
from sklearn.datasets import make_circles, make_classification, make_moons

# 匯入五種分類模型
from sklearn.linear_model import LogisticRegression  # 邏輯斯迴歸
from sklearn.neighbors import KNeighborsClassifier   # K 最近鄰分類
from sklearn.tree import DecisionTreeClassifier      # 決策樹
from sklearn.ensemble import RandomForestClassifier  # 隨機森林
from sklearn.svm import SVC                          # 支援向量機

# 匯入用於分割訓練/測試資料的工具
from sklearn.model_selection import train_test_split

# 匯入用於繪製分類邊界的工具
from sklearn.inspection import DecisionBoundaryDisplay

# 建立包含三種資料集的清單
datasets = [
    make_moons(noise=0.3),                       # 產生半月形的資料，加入 0.3 噪音
    make_circles(noise=0.2, factor=0.5),         # 產生兩個圓形資料集，半徑比例為 0.5，並加入 0.2 噪音
    make_classification(n_features=2, n_redundant=0)  # 產生線性可分的隨機分類資料，2 個特徵，無多餘特徵
]

# 建立一組字典，包含模型名稱對應的模型實例
models = {
    "Logistic Regression": LogisticRegression(),            # 邏輯斯迴歸
    "K Nearest Neighbor": KNeighborsClassifier(n_neighbors=3),  # KNN，k 設為 3
    "Decision Tree": DecisionTreeClassifier(),              # 決策樹
    "Random Forest": RandomForestClassifier(),              # 隨機森林
    "Support Vector Machine": SVC(),                        # 支援向量機
}

# 建立分類用的顏色地圖（紅色代表一類，藍色代表另一類）
colors = ListedColormap(["#FF0000", "#0000FF"])

# 設定圖像中每個子圖的寬度（英吋）
width = 2.5

# 建立整個圖像，包含多列多欄（行數為模型數量 + 原始圖，列數為資料集數量）
figure = plt.figure(figsize=(width * (len(models) + 1), width * len(datasets)))

# 初始化子圖索引編號
i = 0

# 針對每個資料集進行處理與繪圖
for _, data in enumerate(datasets):
    X, y = data  # 將特徵與標籤分開

    # 將資料分為訓練集與測試集，測試集比例為 30%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # 計算 x 和 y 軸的最小最大值，用於設定圖的顯示範圍
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # 子圖編號加 1（每一列的第一個子圖是原始資料）
    i = i + 1
    ax = plt.subplot(len(datasets), len(models) + 1, i)  # 新增子圖位置

    # 繪製訓練資料點
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=colors, edgecolors="k")

    # 繪製測試資料點（較透明）
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=colors, edgecolors="k", alpha=0.6)

    # 設定子圖的顯示範圍與刻度
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())  # 不顯示 x 軸刻度
    ax.set_yticks(())  # 不顯示 y 軸刻度

    # 對每個模型進行訓練並畫出邊界
    for name, model in models.items():
        clf = model.fit(X_train, y_train)  # 使用訓練資料訓練模型
        score = clf.score(X_test, y_test)  # 計算模型在測試集的準確率

        i = i + 1  # 子圖編號加 1
        ax = plt.subplot(len(datasets), len(models) + 1, i)  # 新增子圖

        # 再次繪製訓練與測試資料點
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=colors, edgecolors="k")
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=colors, edgecolors="k", alpha=0.6)

        # 設定子圖的範圍與無刻度
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())

        # 使用 sklearn 的工具繪製模型的分類邊界
        DecisionBoundaryDisplay.from_estimator(
            clf,        # 訓練好的模型
            X,          # 全部的資料特徵（用於描邊界）
            cmap=plt.cm.RdBu,  # 邊界色彩
            alpha=0.8,  # 邊界透明度
            ax=ax,      # 指定繪製的子圖
            eps=0.5     # 決策邊界邊緣留白距離
        )

        # 在右下角顯示模型分數
        ax.text(x_max, y_min, "score: %.2f" % score, size=10, horizontalalignment="right")

        # 如果在第一列，顯示模型的標題
        if i <= len(models) + 1:
            ax.set_title(name)

# 最後將所有圖表一起顯示出來
plt.show()
