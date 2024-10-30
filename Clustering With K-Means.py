##聚类分析是根据数据在坐标上各点的距离远近来分类的-----‘birds of a feather flock together'
#对于单一真值特征，聚类相当于分箱处理或特征离散化；
#对于多个特征，聚类即多维分箱，也叫做向量量化
#聚类算法有很多，不同之处在于如何测量"similarity" or "proximity"和可用于哪种特征
#k-means聚类用欧氏距离来测量各点之间的相似性，先自定有k个中心点centroid，再根据各点与中心点的距离来聚类

#k-means()在scikit-learn包中，主要关注三个参数n_clusters, max_iter, and n_init
#是一个简单的两步程序，算法先根据‘n_clusters’即聚类中心点数k，来分配点到最距离近的群；然后移动各中心点来最小化中心点与各点之间的距离
#这个拟合步骤将迭代到中心点停止移动或者达到设定的最大迭代数'max_iter'
#算法会根据'n_init'重复多次聚类（中心点不同），返回具有总距离值最小的那个最优聚类


#%%例子 加州房产California Housing
#这个数据集中的'Latitude'和'Longitude'是空间特征，将它俩与'MedInc'(median income) 聚类来查看加州不同地区的经济状况

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

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

df = pd.read_csv("C:/Users/Jade Tong/Desktop/KAGGLE/4. Feature Engineering/FE datasets/housing.csv")
X = df.loc[:, ["MedInc", "Latitude", "Longitude"]]
X.head()

#%%   Create cluster feature
kmeans = KMeans(n_clusters=6)  #不设置则默认'n_init'为10
X["Cluster"] = kmeans.fit_predict(X)
X["Cluster"] = X["Cluster"].astype("category")

X.head()

#画散点图来看下聚类效果
sns.relplot(x="Longitude", y="Latitude", hue="Cluster", data=X, height=6)

#%% 
#数据集中的目标是'MedHouseVal'(median house value)，对这个特征画箱型图来查看聚类后各类点的分布情况；
#如果聚类效果好，那么从数据分布中可以看出不同分类的房价中位数应该相差很大
X["MedHouseVal"] = df["MedHouseVal"]
sns.catplot(x="MedHouseVal", y="Cluster", data=X, kind="boxen", height=6)












