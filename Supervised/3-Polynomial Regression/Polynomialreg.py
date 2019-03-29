# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:18:48 2019

@author: harshvardhan
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data=pd.read_csv("Income_Data.csv")
reg=LinearRegression()
polyf=PolynomialFeatures(degree=6)
features=data.iloc[:,0:1]
labels=data.iloc[:,1:]


features_poln = polyf.fit_transform(features)
polyf.fit(features_poln)
reg.fit(features_poln,labels)

""" we will have to change the value too in the polynomial regression 
format in order to predict the value """


print (reg.predict(polyf.fit_transform([[6.5]])))


""" OBSERVING THE RESULT OF POLYNOMIAL REGRESSION """

plt.scatter(features, labels, color = 'red')
plt.plot(features, reg.predict(polyf.fit_transform(features)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()






# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
x=np.array(features)
features_grid = np.arange(min(x), max(x), 0.1)
features_grid = features_grid.reshape((-1, 1))
plt.scatter(features, labels, color = 'red')
plt.plot(features_grid, reg.predict(polyf.fit_transform(features_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


