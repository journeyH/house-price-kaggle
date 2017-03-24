# -*- coding: utf-8 -*-
'''
在kaggle上找的一个预测房价的数据集来做做
'''
__author__ = 'hudie'

import  pandas as pd
import sys
import  matplotlib.pyplot as plot
import numpy as np
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from prepareData import prepareData


housePriceTrain = pd.read_csv('train.csv')
yTrain = housePriceTrain['SalePrice']
housePriceTrain = housePriceTrain.iloc[:,1:-1] #去掉id
housePriceTest = pd.read_csv('test.csv')
housePriceTest = housePriceTest.iloc[:,1:]  #去掉id
yTest =pd.read_csv('sample_submission.csv')
yTest = yTest.iloc[:,1]
'''
train 和test有相同的attirbute 所以数据预处理的时候可以公用prepareData函数
'''

#查看数据基本信息
sys.stdout.write('head information \n')
print(housePriceTrain.head())
sys.stdout.write('tail information \n')
print(housePriceTrain.tail())
sys.stdout.write('data shape \n')
print(housePriceTrain.shape)
'''
 数据有1460个样本，81个特征，因此可以采用复杂的机器学习方法，例如ensembled method进行回归
 不过在描述性统计分析之后，仍可用线性回归方法进行feature engineering 例如用elestic net找到
 特征的重要性排名
'''
####
#1.feature engineering. extract and assemble features to be used for prediction
####
#描述性统计分析
#variable identification
housePriceTrain.info()
housePriceTrain.get_dtype_counts()
'''
 dtypes: float64(3), int64(33), object(43)
 如果要对housePrice进行回归的话，要将object类按照分类重新编码
 并且在回归之前要进行描述性统计以及 查看空缺值后再进行这一步
'''

#处理missing data
totalMissingData = housePriceTrain.isnull().sum().sort_values(ascending=False)
percentMissingData = (housePriceTrain.isnull().sum()/housePriceTrain.isnull().count()).sort_values(ascending=False)
missingData = pd.concat([totalMissingData,percentMissingData], axis=1, keys=['Total', 'Percent'])
missingData.head(20)

'''
可以看到PoolQC、MiscFeature 、Alley 、Fence、FireplaceQu  、LotFrontage 的缺失值占比最小都有17.8% 
GarageCond 、GarageType 、GarageYrBlt 、GarageFinish 、GarageQual 、BsmtExposure、BsmtFinType2、
BsmtFinType1、BsmtCond 、BsmtQual 、MasVnrArea、MasVnrType 、Electrical的缺失值较小
用中间值填补，但是要等到code category做完再做，因为如果用中间值填补，那填补category类的中间值是什么呢？
'''

'''
好啦，花了这么多功夫将category 编码，现在可以愉快的进行描述性分析啦~
例如剔除异常值
以及用elastic net回归得到features的重要性排名等
'''
###
#2.develop targets for training
###
'''
kaggle要求的是Root Mean Squared Logarithmic Error ---
RMSLE penalizes an under-predicted estimate greater than an over-predicted estimate
但是sklearn gradient boosting里面没有这一loss，所以有时间的话我要自己写一个
gradient boosting 用RMSLE来计算loss，当然也应该想sklearn的gradient boosting
一样保健random forest
'''
#3.train a  model
'''
采用sklearn提供的gradient boosting，sklearn 提供的这个类中包含了random forest
所以模型是gradient boosting和random forest混合
'''

xTrain,housePriceNamesTrain = prepareData(housePriceTrain)
xTest, housePriceNamesTest = prepareData(housePriceTest)
housePriceNames =  housePriceNamesTest & housePriceNamesTrain
xTrain = xTrain[housePriceNames]
xTest = xTest[housePriceNames]
'''
但是不幸的是，test有些category并没有train中出现的value
所以在prepareData之后，model的features个数与input（xtest）个数不一致
我只好对train和test的features求一个交集
#'''
nrowXTrain,ncolXTrain = xTrain.shape
#instantiate model
nEst = 10000
depth = 7
learnRate = 0.013
maxFeatures =  int(ncolXTrain/3) #原始作者推荐random forest 回归用features的三分之一 分类用平方根
subsamp = 0.5
housePriceGBMModel = ensemble.GradientBoostingRegressor(n_estimators=nEst,max_depth=depth, 
            learning_rate=learnRate,max_features=maxFeatures,subsample=subsamp, loss='ls')
housePriceGBMModel.fit(xTrain,yTrain)

#4.assess performance on test data

# compute mse on test set
msError = []
RMSLError = []
predictions =housePriceGBMModel.staged_predict(xTest)
yPrediction = housePriceGBMModel.predict(xTest)


for p in predictions:
    msError.append(mean_squared_error(yTest, p))
    root_mean_square_logarithmic = (10**9)*np.sqrt(np.mean((np.log(p + 1) - np.log(yTest + 1)) ** 2))
    RMSLError.append(root_mean_square_logarithmic)
#print("MSE" )
#print(min(msError))
#print(msError.index(min(msError)))
print("RMSL" )
print(min(RMSLError))
print(RMSLError.index(min(RMSLError)))
print(RMSLError.index(max(RMSLError)))

#plot training and test errors vs number of trees in ensemble
plot.figure()
plot.plot(range(1, nEst + 1), housePriceGBMModel.train_score_,
label='Training Set MSE', linestyle=":")
plot.plot(range(1, nEst + 1), RMSLError, label='Test Set RMSL')
#plot.plot(range(1, nEst + 1), msError, label='Test Set MSE')
plot.legend(loc='upper right')
plot.xlabel('Number of Trees in Ensemble')
plot.ylabel('Root Mean Squared Logarithmic Error')
plot.show()
# Plot feature importance
featureImportance = housePriceGBMModel.feature_importances_
# normalize by max importance
featureImportance = featureImportance / featureImportance.max()
idxSorted = np.argsort(-featureImportance)
MostImportantIdxSorted = idxSorted[:min(maxFeatures,30)]
barPos = np.arange(MostImportantIdxSorted.shape[0]) + .5
plot.barh(barPos, featureImportance[MostImportantIdxSorted], align='center')
plot.yticks(barPos, housePriceNames[MostImportantIdxSorted])
plot.xlabel('Variable Importance')
plot.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
plot.show()

