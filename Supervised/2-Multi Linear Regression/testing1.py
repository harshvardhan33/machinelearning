# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:21:15 2019

@author: harshvardhan
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 05 12:06:19 2018

@author: Kunal
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning) 


dataset = pd.read_csv('Salary_Classification.csv')

features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, -1].values

labelencoder = LabelEncoder()
features[:, 0] = labelencoder.fit_transform(features[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
features = onehotencoder.fit_transform(features).toarray()
features = features[:, 1:]
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)
regressor = LinearRegression()
regressor.fit(features_train, labels_train)
pred = regressor.predict(features_test)


""" BACKWARD ELIMINATION TO KNOW THE MOST EFFECTIVE COLUMNS """

features = np.append(arr = np.ones((30, 1)), values = features, axis = 1)

features_opt = features[:,[0,1,2,3,4,5]]
columnlist=['Constant','Dept_Development','Dept_Training','WorkedHours','Certfification','YearsExepreince']
len_column=features_opt.shape[1]
for i in range(len_column+1):
    regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
    x=np.array(regressor_OLS.pvalues)
    if(x.max() > 0.05):
        drop=x.argmax()
        features_opt=np.delete(features_opt,[drop],axis=1)
        del columnlist[drop]
    else:
        break


print(columnlist)
