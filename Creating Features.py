#当已经识别出有潜质的特征后，我们就要develop，下面学几个common transformation
#会用到4个数据集，分别有不同的特征种类

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

accidents = pd.read_csv("C:/Users/Jade Tong/Desktop/KAGGLE/4. Feature Engineering/FE datasets/accidents.csv")
autos = pd.read_csv("C:/Users/Jade Tong/Desktop/KAGGLE/4. Feature Engineering/FE datasets/autos.csv")
concrete = pd.read_csv("C:/Users/Jade Tong/Desktop/KAGGLE/4. Feature Engineering/FE datasets/concrete.csv")
customer = pd.read_csv("C:/Users/Jade Tong/Desktop/KAGGLE/4. Feature Engineering/FE datasets/customer.csv")

#%%            Mathematical Transforms

#数值型特征之间的关系通常可以用数学公式来描述
#数据集‘Automobile’包含一些描述汽车引擎的特征，研究表明特征之间的关系公式可以用来作为potentially useful new features，
#例如‘stroke ratio’可以用来测量how efficient an engine is versus how performant

autos["stroke_ratio"] = autos.stroke/autos.bore

autos[["stroke", "bore", "stroke_ratio"]].head()

#特征的组合越复杂，建模就越困难，例如这个用来描述引擎的马力的公式‘displacement’
autos["displacement"] = (
    np.pi * ((0.5 * autos.bore) ** 2) * autos.stroke * autos.num_of_cylinders)

#%% 
# 数据可视化通过算法来reshape特征，例如数据集‘US Accidents’里的‘WindSpeed’的分布极度偏态，在下面的取对数处理可以有效地使它正态
# If the feature has 0.0 values, use np.log1p 即(log(1+x)) instead of np.log
accidents["LogWindSpeed"] = accidents.WindSpeed.apply(np.log1p) 

# Plot a comparison
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
sns.kdeplot(accidents.WindSpeed, fill=True, ax=axs[0])
sns.kdeplot(accidents.LogWindSpeed, fill=True, ax=axs[1])


#%%                Counts
#用来描述某种事物是否在场的数据通常成对出现，比如说某种疾病的风险因子，对于这种特征可以用aggregate来汇合
# binary：0/1     Boolean：true/false

# 数据集‘Traffic Accidents’含有indicate交通事故发生地点附近是否有roadway object
roadway_features = ["Amenity", "Bump", "Crossing", "GiveWay",
    "Junction", "NoExit", "Railway", "Roundabout", "Station", "Stop",
    "TrafficCalming", "TrafficSignal"]
accidents["RoadwayFeatures"] = accidents[roadway_features].sum(axis=1) #axis=1, 即对行操作；axis=0，则对列

accidents[roadway_features + ["RoadwayFeatures"]].head(10)

#%%     gt() 即greater than 
#也可以用dataframe's built-in methods to create boolean values
#在数据集‘Concrete’里，是不同水泥配方的原料数量，某些配方缺少一种或多种原料，即原料数值为0；
#对数据行处理即可知道各配方含有多少种原料

components = [ "Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
               "Superplasticizer", "CoarseAggregate", "FineAggregate"]
concrete["Components"] = concrete[components].gt(0).sum(axis=1)

concrete[components + ["Components"]].head()


#%%            Buildig-Up and Breaking-Down Features
#有些特征内容是带有标点符号的复合字符串，我们可以用str.split()来将字符串分割成两个或多个简单字符串
#数据集‘Customer Lifetime Value’的数据来自一家保险公司，现在将‘policy’分割成两个特征‘type’和‘level’
customer[["Type","Level"]]=(customer["Policy"].str.split(" ",expand=True))  #当遇到空格键的时候就分割

customer[["Policy", "Type", "Level"]].head()

#也可以将相互有关联的简单特征合成
autos["make_and_style"] = autos["make"] + "_" + autos["body_style"]
autos[["make", "body_style", "make_and_style"]].head()


#%%             Group Transforms
#将某个特征按照要求来分类
#以州分类来算出平均收入
customer["AverageIncome"] = (customer.groupby("State")["Income"].transform("mean"))
customer[["State", "Income", "AverageIncome"]].head()                                             
#mean()函数是数据框内置的，所以我们可以在transform()中将它作为一个字符使用，同样max、min、median、var、std、count这些也行

#计算各州在此数据集中出现的频率
customer["StateFreq"] = (customer.groupby("State")["State"].transform("count")
                         /customer.State.count())

customer[["State", "StateFreq"]].head()

#%%
# 分割customer集，训练集和验证集各一半
df_train = customer.sample(frac=0.5) 
df_valid = customer.drop(df_train.index)

# 用训练集算出各报销方案的平均报销金额
df_train["AverageClaim"] = df_train.groupby("Coverage")["ClaimAmount"].transform("mean")
#一共有三种报销方案premium，extended和basic

# 将验证集与从训练集中得出的平均报销金额合并
df_valid = df_valid.merge(
    df_train[["Coverage", "AverageClaim"]].drop_duplicates(), #drop_duplicates()将两列中的重复值都删除，剩下三种报销方案的平均报销金额
    on="Coverage",how="left",)

df_valid[["Coverage", "AverageClaim"]].head(10)










