#from gemini

import numpy as np
from sklearn.linear_model import LinearRegression

# 建立範例資料
X = np.array([, , , ])
y = np.dot(X, np.array()) + 3  # 目標變數是特徵的線性組合加上一個常數

# 建立線性迴歸模型並進行訓練
reg = LinearRegression().fit(X, y)

# 評估模型
score = reg.score(X, y)
print(f"模型評分 (R^2): {score}")  # 模型評分，越接近1表示模型擬合得越好

# 檢視係數和截距
coefficients = reg.coef_
intercept = reg.intercept_
print(f"係數: {coefficients}")
print(f"截距: {intercept}")

# 進行預測
new_X = np.array()
predicted_y = reg.predict(new_X)
print(f"預測值: {predicted_y}")

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 載入糖尿病資料集
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# 將資料集分割為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 建立並訓練線性迴歸模型
model = LinearRegression()
model.fit(X_train, y_train)

# 在測試集上評估模型
score = model.score(X_test, y_test)
print(f"測試集上的模型評分 (R^2): {score}")

# 進行預測
y_pred = model.predict(X_test)

# 可視化預測結果與實際值的比較
plt.scatter(y_test, y_pred)
plt.xlabel("實際值")
plt.ylabel("預測值")
plt.title("線性迴歸預測 vs 實際值")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--') # 畫一條完美預測的參考線
plt.show()

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 載入乳癌資料集
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

# 將資料集分割為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# 建立並訓練Logistic迴歸模型
model = LogisticRegression(penalty='l2', C=2.0, solver='liblinear', max_iter=1000)
model.fit(X_train, y_train)

# 在測試集上進行預測
y_pred = model.predict(X_test)

# 評估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"準確度: {accuracy}")
print("分類報告:\n", classification_report(y_test, y_pred, target_names=dataset.target_names))
print("混淆矩陣:\n", confusion_matrix(y_test, y_pred))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# 載入鳶尾花資料集
irisData = load_iris()
X = irisData.data
y = irisData.target

# 將資料集分割為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立KNN分類器，設定鄰居數K為7
knn = KNeighborsClassifier(n_neighbors=7)

# 使用訓練資料訓練模型
knn.fit(X_train, y_train)

# 在測試集上進行預測
y_pred = knn.predict(X_test)

# 計算模型準確度
accuracy = accuracy_score(y_test, y_pred)
print(f"準確度: {accuracy}")

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 載入鳶尾花資料集
iris = load_iris()
X = iris.data
y = iris.target

# 將資料集分割為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=99)

# 建立決策樹分類器
clf = DecisionTreeClassifier(random_state=1)

# 使用訓練資料訓練模型
clf.fit(X_train, y_train)

# 在測試集上進行預測
y_pred = clf.predict(X_test)

# 計算模型準確度
accuracy = accuracy_score(y_test, y_pred)
print(f"準確度: {accuracy}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 載入鳶尾花資料集
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 將資料集分割為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立隨機森林分類器，設定樹的數量為100
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 使用訓練資料訓練模型
classifier.fit(X_train, y_train)

# 在測試集上進行預測
y_pred = classifier.predict(X_test)

# 計算模型準確度
accuracy = accuracy_score(y_test, y_pred)
print(f"準確度: {accuracy * 100:.2f}%")

# 繪製混淆矩陣
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False,
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('混淆矩陣熱力圖')
plt.xlabel('預測標籤')
plt.ylabel('真實標籤')
plt.show()

import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 載入手寫數字資料集
data, labels = load_digits(return_X_y=True)
n_samples, n_features = data.shape
n_digits = np.unique(labels).size
print(f"# digits: {n_digits}; # samples: {n_samples}; # features {n_features}")

# 建立一個包含標準化和K-均值分群的pipeline
kmeans = KMeans(n_clusters=n_digits, n_init='auto', random_state=0)
pipeline = make_pipeline(StandardScaler(), kmeans)

# 訓練模型
pipeline.fit(data)

# 獲取分群標籤
predicted_labels = pipeline.predict(data)

# 評估分群結果
print(f"同質性 (Homogeneity): {metrics.homogeneity_score(labels, predicted_labels):.3f}")
print(f"完整性 (Completeness): {metrics.completeness_score(labels, predicted_labels):.3f}")
print(f"V-measure: {metrics.v_measure_score(labels, predicted_labels):.3f}")

# 可視化降維後的資料和簇中心
reduced_data = PCA(n_components=2).fit_transform(data)
kmeans_reduced = KMeans(init='k-means++', n_clusters=n_digits, n_init='auto', random_state=0)
kmeans_reduced.fit(reduced_data)

plt.figure(figsize=(10, 8))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans_reduced.labels_, cmap='viridis')
plt.scatter(kmeans_reduced.cluster_centers_[:, 0], kmeans_reduced.cluster_centers_[:, 1], marker='X', s=200, color='red', label='簇中心')
plt.title('K-均值分群在降維後的digits資料集上')
plt.xlabel('主成分 1')
plt.ylabel('主成分 2')
plt.legend()
plt.show()

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# 載入鳶尾花資料集
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# 建立PCA模型，設定降至3個主成分
pca = PCA(n_components=3)

# 對資料進行降維
X_reduced = pca.fit_transform(X)

# 可視化降維後的資料
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

scatter = ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=y,
    s=40,
)

ax.set(
    title="鳶尾花資料集的前三個PCA維度",
    xlabel="第一主成分",
    ylabel="第二主成分",
    zlabel="第三主成分",
)

ax.xaxis.set_ticklabels()
ax.yaxis.set_ticklabels()
ax.zaxis.set_ticklabels()

# 添加圖例
ax.legend(*scatter.legend_elements(), title="目標類別")

plt.show()

# 輸出每個主成分解釋的變異比例
print("解釋的變異比例:", pca.explained_variance_ratio_)

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

# 載入手寫數字資料集的前500個樣本
digits = datasets.load_digits()
X = digits.data[:500]
y = digits.target[:500]

# 建立t-SNE模型，設定降至2維
tsne = TSNE(n_components=2, random_state=0, learning_rate='auto', init='pca', perplexity=30)

# 對資料進行降維
X_2d = tsne.fit_transform(X)

# 可視化降維後的資料
target_ids = range(len(digits.target_names))
plt.figure(figsize=(6, 5))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
for i, c, label in zip(target_ids, colors, digits.target_names):
    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
plt.legend()
plt.title('t-SNE在digits資料集上的降維結果')
plt.show()
