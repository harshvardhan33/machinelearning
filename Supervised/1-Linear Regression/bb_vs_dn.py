# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 13:27:31 2019

@author: harshvardhan
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
dataset=pd.read_csv("Bahubali2_vs_Dangal.csv")

"""
reg1=LinearRegression()
reg2=LinearRegression()
dataset=pd.read_csv("Bahubali2_vs_Dangal.csv")

features=dataset.iloc[:,0:1]
labels1=dataset.iloc[:,1:-1]
labels2=dataset.iloc[:,2:]

reg1.fit(features,labels1)
x=reg1.predict([[10]])

reg2.fit(features,labels2)
y=reg2.predict([[10]])

"""
reg=LinearRegression()
features=dataset.iloc[:,0:1]
labels=dataset.iloc[:,1:]

reg.fit(features,labels)
x=reg.predict([[10]])

