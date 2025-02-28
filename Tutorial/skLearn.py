# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:11:20 2025

@author: Silas Hauser
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
# from sklearn.tree import DecisionTreeRegressor

handyDataPath = "train.csv"
handyData = pd.read_csv(handyDataPath)
X_features = handyData.columns.drop("price_range")
X = handyData[X_features]
Y = handyData.price_range

handyModel = DecisionTreeRegressor(random_state = 1)
handyModel.fit(X,Y)