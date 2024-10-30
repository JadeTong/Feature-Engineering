#                    Step 1 - Preliminaries
##     Imports and Configuration
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from pandas.api.types import CategoricalDtype

from category_encoders import MEstimateEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor


# Set Matplotlib defaults
plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc("axes",labelweight="bold",labelsize="large",titleweight="bold",titlesize=14,titlepad=10)

#%%      Data Preprocessing 数据预处理
### load数据
df_train=pd.read_csv('C:/Users/Jade Tong/Desktop/KAGGLE/4. Feature Engineering/Final Project Data/train.csv',index_col='Id')
df_test=pd.read_csv('C:/Users/Jade Tong/Desktop/KAGGLE/4. Feature Engineering/Final Project Data/test.csv',index_col='Id')

df=pd.concat([df_train,df_test])

#%% clean 将数据集中有拼写错误的分类变量数据清理一下
# 特征‘Exterior2nd’中有typo，将"Brk Cmn"改成"BrkComm",'Wd Shng'改成'WdShing'
df.Exterior2nd.unique()
#array(['VinylSd', 'MetalSd', 'Wd Shng', 'HdBoard', 'Plywood', 'Wd Sdng',
#       'CmentBd', 'BrkFace', 'Stucco', 'AsbShng', 'Brk Cmn', 'ImStucc',
#       'AsphShn', 'Stone', 'Other', 'CBlock', nan], dtype=object)

df["Exterior2nd"] = df["Exterior2nd"].replace({"Brk Cmn": "BrkComm",
                                               'Wd Shng':'WdShing'})

#"GarageYrBlt"中有缺失值，那就用房子建造时间填充
df.GarageYrBlt.fillna(df.YearBuilt, inplace=True)

#%% encode 将分类变量编码 注意，虽然`MSSubClass`是数值，但是它其实是分类特征
# nominative (unordered) categorical features
Features_nom = ["MSSubClass", "MSZoning", "Street", "Alley", "LandContour", "LotConfig", 
               "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", 
               "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation", "Heating", 
               "CentralAir", "GarageType", "MiscFeature", "SaleType", "SaleCondition"]

# ordinal (ordered) categorical features
# data_description中写明，有些特征赋值"Po", "Fa", "TA", "Gd", "Ex"，有些特征赋值1-10
five_levels = ["Po", "Fa", "TA", "Gd", "Ex"]
ten_levels = list(range(10))
ordered_levels = {"OverallQual": ten_levels,"OverallCond": ten_levels,"ExterQual": five_levels,
                  "ExterCond": five_levels,"BsmtQual": five_levels,"BsmtCond": five_levels,
                  "HeatingQC": five_levels,"KitchenQual": five_levels,"FireplaceQu": five_levels,
                  "GarageQual": five_levels,"GarageCond": five_levels, "PoolQC": five_levels,
                  "LotShape": ["Reg", "IR1", "IR2", "IR3"],
                  "LandSlope": ["Sev", "Mod", "Gtl"],
                  "BsmtExposure": ["No", "Mn", "Av", "Gd"],
                  "BsmtFinType1": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
                  "BsmtFinType2": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
                  "Functional": ["Sal", "Sev", "Maj1", "Maj2", "Mod", "Min2", "Min1", "Typ"],
                  "GarageFinish": ["Unf", "RFn", "Fin"],
                  "PavedDrive": ["N", "P", "Y"],
                  "Utilities": ["NoSeWa", "NoSewr", "AllPub"],
                  "CentralAir": ["N", "Y"],
                  "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
                  "Fence": ["MnWw", "GdWo", "MnPrv", "GdPrv"]}








