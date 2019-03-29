# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:13:18 2018

@author: Kunal
"""

# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Income_Data.csv')

features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, -1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2,
                   random_state = 40)


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(features_train, labels_train)

regressor.predict(features_test)











# Predicting the Test set results
labels_pred = regressor.predict(features_test)







print regressor.predict(6.5)




# Visualising the Training set results
plt.scatter(features_train, labels_train, color = 'red')
plt.plot(features_train, regressor.predict(features_train), color = 'blue')
plt.title('Income vs ML-Experience (Training set)')
plt.xlabel('ML-Experience')
plt.ylabel('Income')
plt.show()


# Visualising the Test set results
#plt.scatter(features_train, labels_train, color = 'green')
plt.scatter(features_test, labels_test, color = 'red')
plt.plot(features_train, regressor.predict(features_train), color = 'blue')
plt.title('Income vs ML-Experience (Test set)')
plt.xlabel('ML-Experience')
plt.ylabel('Income')
plt.show()



#Model accuracy
print regressor.score(features_test, labels_test)
print regressor.score(features_train, labels_train)

#Do we have case of underfitting?
#Do we have case of overfitting?



#Show the animation using learning rate, cost functions and best fit line
#https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9



