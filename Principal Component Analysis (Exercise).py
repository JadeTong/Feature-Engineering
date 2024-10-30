import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

# Set Matplotlib defaults
plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,)

def apply_pca(X, standardize=True):
    # Standardize
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Create principal components
    pca = PCA()
    X_pca = pca.fit_transform(X)
    # Convert to dataframe
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)
    # Create loadings
    loadings = pd.DataFrame(
        pca.components_.T,  # transpose the matrix of loadings
        columns=component_names,  # so the columns are the principal components
        index=X.columns,  # and the rows are the original features
    )
    return pca, X_pca, loadings


def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    return axs


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


def score_dataset(X, y, model=XGBRegressor()):
    # Label encoding for categoricals
    for colname in X.select_dtypes(["category", "object"]):
        X[colname], _ = X[colname].factorize()
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    score = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_squared_log_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score

df = pd.read_csv("C:/Users/Jade Tong/Desktop/KAGGLE/4. Feature Engineering/FE datasets/ames.csv")

#%%
features = ["GarageArea","YearRemodAdd","TotalBsmtSF","GrLivArea"]
print("Correlation with SalePrice:\n")
print(df[features].corrwith(df.SalePrice))

#Correlation with SalePrice:
    
#GarageArea      0.640138
#YearRemodAdd    0.532974
#TotalBsmtSF     0.632529
#GrLivArea       0.706780

#%% 用PCA来破解这些特征之间的相关关系
X = df.copy()
y = X.pop("SalePrice")
X = X.loc[:, features]

#用上面自己定义的‘apply_pca'来简化步骤
pca, X_pca, loadings = apply_pca(X)
print(loadings)
#                   PC1       PC2       PC3       PC4
#GarageArea    0.541229  0.102375 -0.038470  0.833733
#YearRemodAdd  0.427077 -0.886612 -0.049062 -0.170639
#TotalBsmtSF   0.510076  0.360778 -0.666836 -0.406192
#GrLivArea     0.514294  0.270700  0.742592 -0.332837

#%%
#            1) Interpret Component Loadings 解释主成分载荷系数
#主成分一貌似是概括房子大小的主成分，因为每个特征都有正值的，暗示这个主成分在描述高价房和低价房之间的差距
#主成分三在GarageArea和YearRemodAdd的载荷系数都接近于零，可忽略；这个成分主要关于TotalBsmtSF和GrLivArea，描述了生活空间大但是有小地下室的房子和生活空间小但有大地下室的房子之间的差异

# 目标是用主成分分析来发现新特征以提升模型，一个方法是创造以载荷系数为灵感的特征，另一个方法是把主成分本身作为特征加入数据框
#%%          2) Create New Features
#两种方法
#  1: Inspired by loadings
X = df.copy()
y = X.pop("SalePrice")

X["Feature1"] = X.GrLivArea + X.TotalBsmtSF
X["Feature2"] = X.YearRemodAdd * X.TotalBsmtSF

score = score_dataset(X, y)
print(f"Your score: {score:.5f} RMSLE")

#  2: Uses components
X = df.copy()
y = X.pop("SalePrice")

X = X.join(X_pca)
score = score_dataset(X, y)
print(f"Your score: {score:.5f} RMSLE")

#%% 用PCA来检查异常值，PCA可以展示出原特征中看不出的异常变异
#上面有说到房子生活空间和地下室面积的关系，小房子带面积大的地下室并不常见（异常）

#看下得分分布
sns.catplot(y="value",col="variable",data=X_pca.melt(),
    kind='boxen',sharey=False,col_wrap=2)

#从箱型图看出每个主成分都有极端值，看下是哪些数据有极端值

# You can change PC1 to PC2, PC3, or PC4
component = "PC1"

idx = X_pca[component].sort_values(ascending=False).index
df.loc[idx, ["SalePrice", "Neighborhood", "SaleCondition"] + features]



















