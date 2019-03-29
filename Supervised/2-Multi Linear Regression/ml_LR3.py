# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:46:01 2019

@author: harshvardhan
"""

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
reg= LinearRegression()
data=pd.read_csv("iq_size.csv")

features=data.iloc[:,1:]
labels=data.iloc[:,0:1]

features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=40)
"""features_train_scaled=sc.fit_transform(features_train)
features_test_scaled=sc.fit_transform(features_test)
labels_train_scaled=sc.fit_transform(labels_train)
labels_test_scaled=sc.fit_transform(labels_test)"""
reg.fit(features_train,labels_train)
reg.predict(features_test)



"""
x=[94.81,70,153]
x=np.array(x).reshape(1,-1)
y=reg.predict(x)
print(y)
"""

print (reg.score(features_test, labels_test))
print (reg.score(features_train, labels_train))


