# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:05:49 2019

@author: harshvardhan
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("bluegills.csv")

reg=LinearRegression()
polyreg=LinearRegression()
polyfeat=PolynomialFeatures()

features=data.iloc[:,0:1]
labels=data.iloc[:,1:]


"""USING LINEAR REGRESSION"""

reg.fit(features,labels)
x=reg.predict([[5]])
print("ANSWER FROM LINEAR REGRESSION : ",x)

plt.scatter(features, labels, color = 'red')
plt.plot(features, reg.predict(features), color = 'blue')
plt.title('AGE VS LENGTH(Linear Regression)')
plt.xlabel('Age')
plt.ylabel('Length')
plt.show()


"""USING POLYNOMIAL REGRESSION"""
features_poln=polyfeat.fit_transform(features)
polyfeat.fit(features_poln)
polyreg.fit(features_poln,labels)
print ("ANSWER FROM POLYNOMIAL REGRESIION : ",polyreg.predict(polyfeat.fit_transform([[5]])))

temp  =features.sort_values('age')

plt.scatter(features, labels, color = 'red')
plt.plot(temp, polyreg.predict(polyfeat.fit_transform(temp)), color = 'blue')
plt.title('AGE VS LENGTH (Polynomial Regression)')
plt.xlabel('Age')
plt.ylabel('Length')
plt.show()

polyreg.score(polyfeat.fit_transform(features), labels)
reg.score(features,labels)