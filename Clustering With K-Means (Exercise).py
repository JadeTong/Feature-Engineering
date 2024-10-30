import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

# Set Matplotlib defaults
plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc("axes",labelweight="bold",labelsize="large",titleweight="bold",titlesize=14,titlepad=10,)

def score_dataset(X, y, model=XGBRegressor()):
    # Label encoding for categoricals
    for colname in X.select_dtypes(["category", "object"]):
        X[colname], _ = X[colname].factorize()
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    score = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_log_error",)
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score

# Prepare data
df = pd.read_csv("C:/Users/Jade Tong/Desktop/KAGGLE/4. Feature Engineering/FE datasets/ames.csv")

#%%      1) Scaling Feature 特征缩放，归一化（标准化）即将所选特征的value都缩放到一个大致相似的范围
## 如果特征已经是直接可对比的，那就不用归一化（标准化），譬如测试结果；如果特征不可直接对比，譬如身高体重，则需要归一

#%%      2) Create a Feature of Cluster Labels
#Creating a k-means clustering with the following parameters:
# features: LotArea, TotalBsmtSF, FirstFlrSF, SecondFlrSF,GrLivArea
# number of clusters: 10
# iterations: 10

X = df.copy()
y = X.pop("SalePrice")

features = ['LotArea', 'TotalBsmtSF','FirstFlrSF', 'SecondFlrSF','GrLivArea']

# Standardize
X_scaled = X.loc[:, features]
X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)  #特征标准化

kmeans = KMeans(n_clusters=10,n_init=10, random_state=0)
X["Cluster"] = kmeans.fit_predict(X_scaled)

# 可以将聚类结果画出来
Xy = X.copy()
Xy["Cluster"] = Xy.Cluster.astype("category")
Xy["SalePrice"] = y
sns.relplot(
    x="value", y="SalePrice", hue="Cluster", col="variable",
    height=4, aspect=1, facet_kws={'sharex': False}, col_wrap=3,
    data=Xy.melt(value_vars=features, id_vars=["SalePrice", "Cluster"]))
    ##将Xy行列转换，"SalePrice"和"Cluster"用作标识符变量的列id_vars，所选的五个特征为要取消透视的列，被转换成行

#检验一下将聚类结果作为新特征加入到数据集后的XGBoost模型训练效果（均方根对数误差）
score_dataset(X, y)
#0.14243771254616344


#%%    3) Cluster-Distance Features
# 将聚类距离添加到数据框中，用fit_transform语句来生成

X_cd = kmeans.fit_transform(X_scaled)

# Label features and join to dataset
X_cd = pd.DataFrame(X_cd, columns=[f"Centroid_{i}" for i in range(X_cd.shape[1])])
X = X.join(X_cd)

score_dataset(X, y)
# 0.13822238795813688












