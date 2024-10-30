#将从mutual information练习中得到最大mi值的特征在这个练习中继续挖掘。

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

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

# Prepare data
df = pd.read_csv("C:/Users/Jade Tong/Desktop/KAGGLE/4. Feature Engineering/FE datasets/ames.csv")
X = df.copy()
y = X.pop("SalePrice")

## 从一些mathematical combinations开始，注重于描写区域的特征（都有相同的单位，平方尺）
#%%           1) Create Mathematical Transforms
#生成这些特征：
#1. LivLotRatio: the ratio of GrLivArea to LotArea
#2. Spaciousness: the sum of FirstFlrSF and SecondFlrSF divided by TotRmsAbvGrd
#3. TotalOutsideSF: the sum of WoodDeckSF, OpenPorchSF, EnclosedPorch, Threeseasonporch, and ScreenPorch

X_1 = pd.DataFrame()  # 创建一个新数据框来存放新特征

X_1["LivLotRatio"] = X.GrLivArea / X.LotArea
X_1["Spaciousness"] = (X.FirstFlrSF + X.SecondFlrSF) / X.TotRmsAbvGrd
X_1["TotalOutsideSF"] = X.WoodDeckSF + X.OpenPorchSF + X.EnclosedPorch + X.Threeseasonporch + X.ScreenPorch

#%%           2) Interaction with a Categorical
# 如果你发现数值型特征和类别型特征之间有交互作用，那就需要将类别型特征one-hot encode处理，
# 再将分类变量和连续变量以行相乘，独热编码后false即为0，true即为1，相乘后true的才有不为零的值

# 发现特征BldgType和GrLivArea之间有交互作用，现在生成代表它们之间的交互作用的特征
# 独热编码BldgType. 用前缀`prefix="Bldg"` in `get_dummies`
X_2 = pd.get_dummies(X.BldgType,prefix='Bldg') 
# Multiply
X_2 = X_2.mul(df.GrLivArea, axis=0)


#%%           3) Count Feature
#生成描述居住地有多少种户外区域的特征'PorchTypes'，计算一间屋有以下的几个户外区域，用gt(0)

X_3 = pd.DataFrame()

prochtypes = ['WoodDeckSF','OpenPorchSF','EnclosedPorch','Threeseasonporch','ScreenPorch']
X_3["PorchTypes"] = X[prochtypes].gt(0).sum(axis=1)  #axis=1，即横向聚合，默认为axis=0，纵向聚合


#%%           4) Break Down a Categorical Feature
# MSSubClass描述了房屋的类型，
df.MSSubClass.unique()
#可以看出字符串是由多个单词用_串联一起的，现在只取字符串中第一个单词
X_4 = pd.DataFrame()

X_4["MSClass"] = X.MSSubClass.str.split('_',expand=True,n=1)[0] #n=1，即分割一次就停止

#%%           5) Use a Grouped Transform
#生成特征MedNhbdArea来描述各Neighborhood的GrLivArea中位数
X_5 = pd.DataFrame()

X_5["MedNhbdArea"] = X.groupby('Neighborhood')['GrLivArea'].transform('median')

#%%  最后将各个新特征组合一起
X_new = X.join([X_1, X_2, X_3, X_4, X_5])
X_new

score_dataset(X_new, y)  ##算均方根对数误差













