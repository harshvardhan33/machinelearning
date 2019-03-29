# -*- coding: utf-8 -*-
"""
Created on Mon Mar 05 12:06:19 2018

@author: Kunal
"""

# Multiple Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Classification.csv')
temp = dataset.values
features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
features[:, 0] = labelencoder.fit_transform(features[:, 0])

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
features = onehotencoder.fit_transform(features).toarray()

# Avoiding the Dummy Variable Trap
features = features[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features_train, labels_train)

# Predicting the Test set results
Pred = regressor.predict(features_test)

# Predict the salary of a developer having 3000 worked hours, with 2 certificates and 2 years of exp
le = labelencoder.transform(['Development'])
le = le.reshape(-1,1)
dv = onehotencoder.transform(le).toarray()
dv = dv.reshape(-1,1)
x = [dv[0][1], dv[0][2],3000,2,2]
x = np.array(x)
regressor.predict(x.reshape(1, -1))


x = []

for item in dv[0]:
    x.append(item)


#Just for testing
labelencoder.inverse_transform(0)





# Getting Score for the Multi Linear Reg model
Score = regressor.score(features_train, labels_train)
Score = regressor.score(features_test, labels_test)

# Predicting for some other values
x = np.array([0,	0,1150,	3,	4]).reshape(1,-1)

print regressor.predict(x)
print regressor.coef_













# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
#This is done because statsmodels library requires it to be done for constants.
features = np.append(arr = np.ones((30, 1)), values = features, axis = 1)



features_opt = features[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()



features_opt = features[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()





features_opt = features[:, [0, 1, 3, 5]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()




features_opt = features[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_OLS.summary()



features_opt = features[:, [0, 5]]
regressor_OLS = sm.OLS(endog = labels, exog = features_opt).fit()
print regressor_OLS.summary()

"""
Few comments about OLS for dummy variable values

Case Study
Suppose you are building a linear (or logistic) regression 
model. In your independent variables list, you have a 
categorical variable with 4 categories (or levels). 
You created 3 dummy variables (k-1 categories) and 
set one of the category as a reference category. 
Then you run stepwise / backward/ forward regression 
technique and you found only one of the category coming 
out statistically significant based on p-value and the 
remaining 3 categories are insignificant. 
The question arises - should we remove or keep these 3 
categories having insignificant difference? should we 
include the whole categorical variable or not?

Solution
In short, the answer is we can ONLY choose whether we 
should use this independent categorical variable as a 
whole or not. In other words, we should only see whether 
the categorical variable as a whole is significant or not. 
We cannot include some categories of a variable and exclude 
some categories having insignificant difference.

Ref: https://www.listendata.com/2016/07/insignificant-levels-of-categorical-variable.html
"""

