import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

# Set Matplotlib defaults
plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

#%% 读数
# 数据集包含房价与78个房子特征的数据
df = pd.read_csv("C:/Users/Jade Tong/Desktop/KAGGLE/4. Feature Engineering/FE datasets/ames.csv")
df.head()

#%% 教程中的函数
def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    
#%%                1)understand mutual information
# 先看一下几个特征对应的房价散点图
features = ["YearBuilt", "MoSold", "ScreenPorch"]
sns.relplot(x="value", y="SalePrice", col="variable", 
            data=df.melt(id_vars="SalePrice", value_vars=features), 
            facet_kws=dict(sharex=False),)

#从散点图上看出'YearBuilt'应该与目标房价有高MI值，因为年份倾向于将房价范围控制在一个较小的区间；
#而'MoSold'并不能控制房价

#%% 用make_mi_scores来算出mi值，然后选出mi值高的特征
X = df.copy()
y = X.pop('SalePrice')
mi_scores = make_mi_scores(X, y)

print(mi_scores.head(20)) #前20个
print(mi_scores.tail(20)) #后20个

plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores.head(20))  #前20个
plot_mi_scores(mi_scores.tail(20))  #后20个

#%%                2)examine mi scores
# 调查一下特征'BldgType',它的MI值并不高，即它对房价影响并不大，这也可以从箱型图中看出
sns.catplot(x="BldgType", y="SalePrice", data=df, kind="boxen")

#%%
# 但是再看下它跟‘GrLivArea’（Above ground living area）和‘MoSold’（Month sold）的交互作用
sns.lmplot(x="GrLivArea", y="SalePrice", hue="BldgType", col="BldgType",
           data=df, scatter_kws={"edgecolor": 'w'}, col_wrap=3, height=4,)

sns.lmplot(x="MoSold", y="SalePrice", hue="BldgType", col="BldgType",
           data=df, scatter_kws={"edgecolor": 'w'}, col_wrap=3, height=4,)

#BldgType对GrLivArea有显著影响，对MoSold没有
















