# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 12:54:40 2019

@author: harshvardhan
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



data=pd.read_csv("Foodtruck.csv")
reg=LinearRegression()

features=data.iloc[:,0:1]
labels=data.iloc[:,1:]


features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size = 0.2,random_state = 40)
reg.fit(features_train,labels_train)

x=reg.predict([[3.073]])
print(x)



plt.scatter(features_train, labels_train, color = 'red')
plt.plot(features_train, reg.predict(features_train), color = 'blue')
plt.title('Population vs Profit(Training set)')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.show()




plt.scatter(features_test, labels_test, color = 'red')
plt.plot(features_train, reg.predict(features_train), color = 'blue')
plt.title('Population vs Profit(Test set)')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.show()
