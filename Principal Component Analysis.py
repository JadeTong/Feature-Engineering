# 聚类分析是根据点距离对数据分区，而主成分分析是根据数据的变异程度来分类
# 目的是把多个变量压缩为少数几个综合指标（称为主成分），使得综合指标能够包含原来的多个变量的主要的信息。
# 主成分分析是帮助探寻数据间关系的工具，也可以用来生成更有信息量的特征
# note：PCA一般应用在标准化数据上，因为对于标准化数据，变异'variation'代表相关性'correlation',
      # 但是，对于未标准化数据，"variation" means "covariance"协方差


#%%       Principal Component Analysis
# the idea of PCA: instead of describing the data with the original features, 
# we describe it with its axes of variation.

# There will be as many principal components as there are features in the original dataset
# 主成分可用加权过（载荷系数）的特征来表达

#PCA只在数值型特征上使用
#PCA对数据的分布很敏感，应用PCA前因先标准化数据
#异常值对影响分析结果，所以要考虑剔除或控制异常值

#%% 例子 1985 Automobiles
#加载数据
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.feature_selection import mutual_info_regression

plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc("axes",labelweight="bold",labelsize="large",titleweight="bold",titlesize=14,titlepad=10,)

df = pd.read_csv("C:/Users/Jade Tong/Desktop/KAGGLE/4. Feature Engineering/FE datasets/autos.csv")

#算出与目标特征‘price’的MI值
def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0))
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0))
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    return axs

#%% 
#从Mutual Information那节中知道"highway_mpg", "engine_size", "horsepower", "curb_weight"这几个特征的MI值较高
#先标准化这些特征，使它们在同一个scale上
features = ["highway_mpg", "engine_size", "horsepower", "curb_weight"]
X = df.copy()
y = X.pop('price')
X = X.loc[:, features]

# Standardize
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

#%%  PCA
from sklearn.decomposition import PCA

# Create principal components
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Convert to dataframe
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)

X_pca.head()      ##这些是主成分得分
#        PC1       PC2       PC3       PC4
#0  0.382486 -0.400222  0.124122  0.169539
#1  0.382486 -0.400222  0.124122  0.169539
#2  1.550890 -0.107175  0.598361 -0.256081
#3 -0.408859 -0.425947  0.243335  0.013920
#4  1.132749 -0.814565 -0.202885  0.224138

#%%   载荷系数loadings

#当完成fitting这步后，主成分分析就会将载荷系数放在‘components_’attribute上，即pca.components_
loadings = pd.DataFrame(
    pca.components_.T,  # ‘T’将矩阵转置
    columns=component_names,  # 将列名设为主成分名
    index=X.columns)  # 将行名设为原特征名
loadings
#                  PC1       PC2       PC3       PC4
#highway_mpg -0.492347  0.770892  0.070142 -0.397996
#engine_size  0.503859  0.626709  0.019960  0.594107
#horsepower   0.500448  0.013788  0.731093 -0.463534
#curb_weight  0.503262  0.113008 -0.678369 -0.523232

#%% 解释方差和累积方差
plot_variance(pca)


#%% 将各主成分作为特征，计算与目标‘price’的MI值
mi_scores = make_mi_scores(X_pca, y, discrete_features=False)
mi_scores
# PC1    1.015315
# PC2    0.379752
# PC3    0.307075
# PC4    0.204475

#%%  主成分三在horsepower和curb_weight之间呈反向关系，看起来像是sports cars与wagons之间的对抗
# 看看按主成分三得分的排列顺序的品牌和车型数据
idx = X_pca["PC3"].sort_values(ascending=False).index   #顺序
cols = ["make", "body_style", "horsepower", "curb_weight"]
df.loc[idx, cols]

#%%
df["sports_or_wagon"] = X.curb_weight / X.horsepower
sns.regplot(x="sports_or_wagon", y='price', data=df, order=2);






