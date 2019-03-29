# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:39:29 2019

@author: harshvardhan
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap


dataset=pd.read_csv("Social_Network_Ads.csv")
features=dataset.iloc[:,[2,3]].values
labels=dataset.iloc[:,4].values

features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.25,random_state=0)
sc=StandardScaler()
features_train=sc.fit_transform(features_train)
features_test=sc.transform(features_test) 

classifier=LogisticRegression()
classifier.fit(features_train,labels_train)
labels_pred=classifier.predict(features_test)


cm=confusion_matrix(labels_test,labels_pred)


"""
features_set, labels_set = features_train, labels_train
features1, features2 = np.meshgrid(np.arange(start = features_set[:, 0].min() - 1, stop = features_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = features_set[:, 1].min() - 1, stop = features_set[:, 1].max() + 1, step = 0.01))
plt.contourf(features1, features2, classifier.predict(np.array([features1.ravel(), features2.ravel()]).T).reshape(features1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(features1.min(), features1.max())
plt.ylim(features2.min(), features2.max())
for i, j in enumerate(np.unique(labels_set)):
    plt.scatter(features_set[labels_set == j, 0], features_set[labels_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()




features_set, labels_set = features_test, labels_test
features1, features2 = np.meshgrid(np.arange(start = features_set[:, 0].min() - 1, stop = features_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = features_set[:, 1].min() - 1, stop = features_set[:, 1].max() + 1, step = 0.01))
plt.contourf(features1, features2, classifier.predict(np.array([features1.ravel(), features2.ravel()]).T).reshape(features1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(features1.min(), features1.max())
plt.ylim(features2.min(), features2.max())
for i, j in enumerate(np.unique(labels_set)):
    plt.scatter(features_set[labels_set == j, 0], features_set[labels_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

"""

"""DOING IT WITH EASIER CODE AND BETTER APPROACH"""


x_min, x_max = features_train[:, 0].min() - 1, features_train[:, 0].max() + 1
y_min, y_max = features_train[:, 1].min() - 1, features_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)



plt.plot(features_test[labels_test == 1, 0], features_test[labels_test == 1, 1], 'bo', label='Class 2')
plt.plot(features_test[labels_test == 0, 0], features_test[labels_test == 0, 1], 'ro', label='Class 1')

plt.contourf(xx, yy, Z, alpha=1.0)
plt.show()

