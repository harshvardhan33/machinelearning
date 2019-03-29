# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:50:26 2019

@author: harshvardhan
"""

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
from sklearn.compose import ColumnTransformer

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



dataset=pd.read_csv("mushrooms.csv")
features=dataset.iloc[:,[5,21,22]].values
labels=dataset.iloc[:,0].values.reshape(-1,1)

transformer=ColumnTransformer(sparse_threshold=0,transformers=[("OneHot",OneHotEncoder(),[0,1,2])])
features=transformer.fit_transform(features)

features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.25,random_state=0)

classifier = KNeighborsClassifier(n_neighbors = 5, p = 2)
classifier.fit(features_train,labels_train)
labels_pred=classifier.predict(features_test)

print (classifier.score(features_test, labels_test))
print (classifier.score(features_train, labels_train))
