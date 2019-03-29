# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:19:52 2019

@author: harshvardhan
"""

import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



dataset=pd.read_csv("mushrooms.csv")
features=dataset.iloc[:,[5,21,22]].values
labels=dataset.iloc[:,0].values.reshape(-1,1)
lenc=LabelEncoder()

features[:,0] = lenc.fit_transform(features[:,0])
features[:,1] = lenc.fit_transform(features[:,1])
features[:,2] = lenc.fit_transform(features[:,2])
labels[:,0] =lenc.fit_transform(labels[:,0])

onehotencoder = OneHotEncoder(categorical_features = [0,1,2])
onehotencoder1 = OneHotEncoder(categorical_features = [0])
features= onehotencoder.fit_transform(features).toarray()
labels= onehotencoder1.fit_transform(labels).toarray()

features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.25,random_state=0)

classifier = KNeighborsClassifier(n_neighbors = 5, p = 2)
classifier.fit(features_train,labels_train)
labels_pred=classifier.predict(features_test)


print (classifier.score(features_test, labels_test))
print (classifier.score(features_train, labels_train))
