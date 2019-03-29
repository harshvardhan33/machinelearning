# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:07:23 2019

@author: harshvardhan
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

reg=LinearRegression()
dataset=pd.read_csv("Income_Data.csv")
#identifying feature and label column and seperating them into new dataframes
features=dataset.iloc[:,:-1].values
label=dataset.iloc[:,-1:]


#Making test and train values 
features_train,features_test,label_train,label_test=train_test_split(features, label, test_size = 0.2,random_state = 40)
#Using the fit method to train 

reg.fit(features_train,label_train)

x=reg.predict([[1.5]]).reshape(1,)

plt.scatter(features_train, label_train, color = 'red')
plt.plot(features_train, reg.predict(features_train), color = 'blue')
plt.title('Income vs ML-Experience (Training set)')
plt.xlabel('ML-Experience')
plt.ylabel('Income')
plt.show()



plt.scatter(features_test, label_test, color = 'red')
plt.plot(features_train, reg.predict(features_train), color = 'blue')
plt.title('Income vs ML-Experience (Test set)')
plt.xlabel('ML-Experience')
plt.ylabel('Income')
plt.show()





print (reg.score(features_test, label_test))
print (reg.score(features_train, label_train))








