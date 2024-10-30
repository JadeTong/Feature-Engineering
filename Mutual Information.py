#%% 特征选择之互信息（mutual information）
##  数据集‘autos’含有来自1985年的193辆车的数据，目标是从车辆的23个特征中预测出车的价格，
##  在这个例子中，会用mutual information来评价特征并用数据可视化来研究预测结果

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")

df = pd.read_csv("C:/Users/Jade Tong/Desktop/KAGGLE/4. Feature Engineering/FE datasets/autos.csv")
df.head()

#%% scikit-learn包中的MI算法对于离散型特征和连续型特征的处理不一样，所以要先说明特征类型
X = df.copy()
y = X.pop("price")

# Label encoding for categoricals
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()   ###factorize()将分类变量编码为数值型

# All discrete features should now have integer dtypes (double-check this before using MI!)
discrete_features = X.dtypes == int

#%% 算出MI值
from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)
mi_scores[::3]  # show a few features with their MI scores

#%%  画个条形图更直观
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)
#从条形图中看出curb_weight跟价格之间有强关系

#%% 对curb_weight和price画一个散点图
sns.relplot(x="curb_weight", y="price", data=df)

# fuel_type有个相对低的MI值，但是两种不同燃料对马力（horsepower）有不同影响，所以说要删掉一个MI值低的特征前，先调查一下它是否有交互作用。
sns.lmplot(x="horsepower", y="price", hue="fuel_type", data=df);



























