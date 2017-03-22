# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 21:19:49 2017

@author: hudie
"""
import  pandas as pd
import numpy as np

def prepareData(housePrice):
    delColumn = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage']
    for column in delColumn: del housePrice[column]
#    label = housePrice['SalePrice']
    #code category features
    findObject = housePrice.dtypes == np.object
    #numerical features
    remainfeatures = housePrice.loc[:,~findObject]
#    remainfeatures = remain.iloc[:,:-1]
    #category features
    objectNeedCode = housePrice.loc[:,findObject]
    nrow, ncol = objectNeedCode.shape
    allNewFeatures = pd.DataFrame([])
    categoryValueList = [] #为了方便分析对应code的feature的原始类别值
    for col in range(ncol):
        unique = set(objectNeedCode.iloc[:,col])
        #print(unique)
        #根据set的种类code 
        uniqueLen = len(unique)
        featureCoded = []
        featureName = []
        newFeatures = []
        uniqueColumnName = []
        uniqueList = list(unique)
        categoryValueList.append(uniqueList)
        for uniqueId in range(uniqueLen):
            for row in range(nrow):
                codeCol = [0.0]*uniqueLen
                if objectNeedCode.iloc[row,col] == uniqueList[uniqueId]:
                    codeCol[uniqueId] = 1.0
                    featureCoded.append(codeCol)
                else:
                    continue       
            uniqueColumnName = objectNeedCode.columns[col] + str(uniqueId)
            featureName.append(uniqueColumnName)
        newFeatures = pd.DataFrame(featureCoded,  columns=featureName)
        allNewFeatures = pd.concat([allNewFeatures, newFeatures],axis = 1)
    
    features = pd.concat([remainfeatures, allNewFeatures], axis = 1)
    
    #fill missing data
    meanVals = features.mean()
    features.fillna(meanVals,inplace=True)
    housePriceNames = features.columns
    return features,housePriceNames