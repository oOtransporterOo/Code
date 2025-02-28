# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:10:31 2025

@author: Silas Hauser
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
# warnings.filterwarnings('ignore')

df_train = pd.read_csv("./house/train.csv")

def plotRelationship():
    data = pd.concat([df_train['SalePrice'], df_train.GrLivArea], axis=1)
    data.plot.scatter(x="GrLivArea", y='SalePrice', ylim=(0,800000));
    
def plotBoxPlot():
    var = 'OverallQual'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=var, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000);

def correlationMatrix():
    corrmat = df_train.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True);


def correlationMatrixLargestCorrelation():
    k = 10 #number of variables for heatmap
    corrmat = df_train.corr(numeric_only=True)
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(df_train[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
    
def scatterPlot():
    sns.set()
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    sns.pairplot(df_train[cols], size = 2.5)
    plt.show();
    
def missingData():
    total = df_train.isnull().sum().sort_values(ascending=False)
    percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data

def dropMissingData():
    df_trainNew = df_train.drop((missingData()[missingData()['Total'] > 1]).index,axis=1,inplace=True)
    df_trainNew = df_trainNew.drop(df_trainNew.loc[df_trainNew['Electrical'].isnull()].index)
    df_trainNew.isnull().sum().max() #just checking that there's no missing data missing...


# histogramm and normal probaility plot
def plotHistoAndProb():
    sns.distplot(df_train['SalePrice'], fit=norm);
    fig = plt.figure()
    res = stats.probplot(df_train['SalePrice'], plot=plt)
    
#applying log transformation
def logTransform():
    df_train['SalePrice'] = np.log(df_train['SalePrice'])
  
# transforming Data with zeros
def transformWithZero():
    #create column for new variable (one is enough because it's a binary categorical feature)
    #if area>0 it gets 1, for area==0 it gets 0
    df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
    df_train['HasBsmt'] = 0 
    df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
    #transform data
    df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
    #histogram and normal probability plot
    sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
    fig = plt.figure()
    res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
    
# each non numeric variable is converted into as many 0/1 variables as there are different values
def getMatrixForNonNumeric():
    return pd.get_dummies(df_train)