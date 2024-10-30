import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from category_encoders import MEstimateEncoder
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

# Set Matplotlib defaults
plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc("axes",labelweight="bold",labelsize="large",titleweight="bold",titlesize=14,titlepad=10)

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

#%% 选哪个特征来做目标特征，一般选种类多的分类变量
df.select_dtypes(["object"]).nunique()
#%% ‘Neighborhood'最多种类，看看每个种类分别出现多少次
df["Neighborhood"].value_counts()

#%% 避免过拟合，我们需要切割一部分数据来fit the encoder
# Encoding split
X_encode = df.sample(frac=0.20, random_state=0)
y_encode = X_encode.pop("SalePrice")

# Training split
X_pretrain = df.drop(X_encode.index)
y_train = X_pretrain.pop("SalePrice")

#%%     应用M-Estimate Encoding
encoder = MEstimateEncoder(cols=['Neighborhood'],m=5)

# Fit the encoder on the encoding split
encoder.fit(X_encode,y_encode)

# Encode the training split
X_train = encoder.transform(X_pretrain, y_train)

#%%
plt.figure(dpi=90)
ax = sns.distplot(y_train, kde=False, norm_hist=True)
ax = sns.kdeplot(X_train.Neighborhood, color='r', ax=ax)
ax.set_xlabel("SalePrice")
ax.legend(labels=['Neighborhood', 'SalePrice'])

#%%
X = df.copy()
y = X.pop("SalePrice")
score_base = score_dataset(X, y)
score_new = score_dataset(X_train, y_train)

print(f"Baseline Score: {score_base:.4f} RMSLE")
print(f"Score with Encoding: {score_new:.4f} RMSLE")
# Baseline Score: 0.1434 RMSLE
# Score with Encoding: 0.1398 RMSLE

#编码后 XGBoost模型效果并没有优化很多，很有可能是因为the extra information gained by the encoding couldn't make up for the loss of data used for the encoding.

#%%  探索target encoding带来的过拟合
# 假设我们新建一个特征，它的数值就单纯是顺序：0、1、2.....
# m定为0，就做一个mean-encode
X = df.copy()
y = X.pop('SalePrice')

X["Count"] = range(len(X))
X["Count"][1] = 0  # actually need one duplicate value to circumvent error-checking in MEstimateEncoder

encoder = MEstimateEncoder(cols="Count", m=0)
X = encoder.fit_transform(X, y)

score =  score_dataset(X, y)
print(f"Score: {score:.4f} RMSLE")
#Score: 0.0375 RMSLE   几乎为零的误差值！

#画个分布图来看看
plt.figure(dpi=90)
ax = sns.distplot(y, kde=True, hist=False)
ax = sns.distplot(X["Count"], color='r', ax=ax, hist=True, kde=False, norm_hist=True)
ax.set_xlabel("SalePrice")
#mean-encode后的count也和saleprice分布差不多

#count没有重复值，均值编码后的count就是saleprice的复制，也就是说通过均值编码，一个无意义的特征变成了一个完美的特征
#那是因为没有事先分割数据集来训练编码，使编码过程和xgboost过程用了同一个数据集，如果事先切割的数据，那这种情况就不会出现





