#跟one-hot encoding和标签编码同理，但利用目标来帮助编码
#A target encoding is any kind of encoding that replaces a feature's categories with some number derived from the target.

import pandas as pd

autos = pd.read_csv("C:/Users/Jade Tong/Desktop/KAGGLE/4. Feature Engineering/FE datasets/autos.csv")
#%%        mean encoding

#最简单的方法是聚合处理如均值
autos["make_encoded"] = autos.groupby("make")["price"].transform("mean")
autos[["make", "price", "make_encoded"]].head(10)

#如果应用于二元目标，那就叫bin counting分箱计数

#%%        Smoothing 目标编码会造成过拟合
# First are unknown categories, 
# Target encodings create a special risk of overfitting, which means they need to be trained on an independent "encoding" split.
# When you join the encoding to future splits, Pandas will fill in missing values for any categories not present in the encoding split. These missing values you would have to impute somehow.

#Second are rare categories. When a category only occurs a few times in the dataset, any statistics calculated on its group are unlikely to be very accurate.

# The idea of smoothing is to blend the in-category average with the overall average

# In pseudocode:
# encoding = weight * in_category avg + (1 - weight) * overall avg

# An easy way to determine the value for weight is to compute an m-estimate:
# weight = n / (n + m)  n系某种类出现次数，m越大，总体均值的权重越大

#在Automobiles这个数据集里面，有3辆车来自品牌Chevrolet，如果定m=2，那么w=3/(3+2)=0.6，即
# chevrolet = 0.6 * 6000.00 + 0.4 * 13285.03

----------Use Cases for Target Encoding
Target encoding is great for:
High-cardinality features: A feature with a large number of categories can be troublesome to encode: a one-hot encoding would generate too many features and alternatives, like a label encoding, might not be appropriate for that feature. A target encoding derives numbers for the categories using the feature's most important property: its relationship with the target.
Domain-motivated features: From prior experience, you might suspect that a categorical feature should be important even if it scored poorly with a feature metric. A target encoding can help reveal a feature's true informativeness.

#%% 例子  MovieLens1M
# MovieLens1M数据集含有来自MovieLens网站的一百万条电影评分

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc("axes",labelweight="bold",labelsize="large",titleweight="bold",titlesize=14,titlepad=10)

df = pd.read_csv("C:/Users/Jade Tong/Desktop/KAGGLE/4. Feature Engineering/FE datasets/movielens1m.csv")
df = df.astype(np.uint8, errors='ignore') # reduce memory footprint
print("Number of Unique Zipcodes: {}".format(df["Zipcode"].nunique()))
#Number of Unique Zipcodes: 3439
df.head()
#特征‘zipcode'有超过3000个类别，使它成为目标编码的优秀选择，而且数据集体量大，意味着可以切一部分来训练编码

#%% split 25%来训练目标编码
X = df.copy()
y = X.pop('Rating')

X_encode = X.sample(frac=0.25)     #编码集
y_encode = y[X_encode.index]

X_pretrain = X.drop(X_encode.index)   #训练集
y_train = y[X_pretrain.index]

#%%  scikit-learn-contrib中的一个包category_encoders内置了一个m-estimate encoder
from category_encoders import MEstimateEncoder

# Create the encoder instance. Choose m to control noise.
encoder = MEstimateEncoder(cols=["Zipcode"], m=5.0)   #定m为5

# Fit the encoder on the encoding split.
encoder.fit(X_encode, y_encode)

# Encode the Zipcode column to create the final training data
X_train = encoder.transform(X_pretrain)

#%% 对比编码过的值和目标y值来看下编码过后的特征有多informative

plt.figure(dpi=90)
ax = sns.distplot(y, kde=False, norm_hist=True)
ax = sns.kdeplot(X_train.Zipcode, color='r', ax=ax)
ax.set_xlabel("Rating")
ax.legend(labels=['Zipcode', 'Rating'])

#编码过后的'zipcode'roughly follows真实ratings的分布，说明来自不同zipcode的观众对评分的差异足够大，让我们的target encoding捕捉到有效信息



















