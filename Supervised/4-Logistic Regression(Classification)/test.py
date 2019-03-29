# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:53:02 2019

@author: harshvardhan
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


reg=LinearRegression()
dataset=pd.read_csv("Salary_Classification.csv")
features=dataset.iloc[:,0:-1].values
labels=dataset.iloc[:,-1:].values

transformer=ColumnTransformer(transformers=[("OneHot",OneHotEncoder(),[0])])
onehot=transformer.fit_transform(features)


