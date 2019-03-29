# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:30:43 2019

@author: harshvardhan
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap



dataset=pd.read_csv("Social_Network_Ads.csv")
features=dataset.iloc[:,[2,3]].values
labels=dataset.iloc[:,4].values


features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.25,random_state=0)
sc=StandardScaler()
features_train=sc.fit_transform(features_train)
features_test=sc.transform(features_test)


"""NOW WE WILL USE THE K NEIGHBOUR CLASSIFIER ,HERE WE HAVE TO SPECIFY THAT 
WHAT ARE THE NUMBER IOG NEIGHBOURS I WANT TO CONSIDER AND WHAT IS THE TYPE OF 
DISTANCE FORMULA I WANT TO USE 
WHEN P = 1 it is treated as manhattan distance  
WHEN P = 2 it is treated as eucledian distance """
classifier=KNeighborsClassifier(n_neighbors=5,p=2)
classifier.fit(features_train, labels_train)
labels_pred = classifier.predict(features_test)


"""CREATING A CONFUSION MATRIX """
cm = confusion_matrix(labels_test, labels_pred)

"""making the range or values from x and y column and keeping a margin of 
+1 and -1 """

x_min,x_max=features_train[:, 0].min() - 1, features_train[:, 0].max() + 1
y_min,y_max=features_train[:, 1].min() - 1, features_train[:, 1].max() + 1

"""making a meshgrid of all the values in order to create all possible combinations """

"""use the np.arrange method in order to create the grid of values seperated by 0.1 distance"""
xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))


"""use the ravel method to convert the matrix form into the 1D series in order to
concatenate 1 from each row (x and y )
(x0,y0),(x1,y1)...and so on """

z=classifier.predict(np.c_[xx.ravel(),yy.ravel()])
z=z.reshape(xx.shape)

"""Plotting the final results """

plt.plot(features_test[labels_test == 1, 0], features_test[labels_test == 1, 1], 'bo', label='Class 2')
plt.plot(features_test[labels_test == 0, 0], features_test[labels_test == 0, 1], 'ro', label='Class 1')


plt.contourf(xx, yy, z, alpha=1.0)
plt.show()


print (classifier.score(features_test, labels_test))
print (classifier.score(features_train, labels_train))
