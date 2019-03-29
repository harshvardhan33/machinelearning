# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:36:20 2019

@author: harshvardhan
"""

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore", category=FutureWarning) 
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


data=pd.read_csv("affairs.csv")
features=data.iloc[:,:-1].values
labels=data.iloc[:,-1:].values


features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.25,random_state=0)

classifier=LogisticRegression()
classifier.fit(features_train,labels_train)
labels_pred=classifier.predict(features_test)
"""
print (classifier.score(features_test, labels_test))
print (classifier.score(features_train, labels_train))
"""
cm=confusion_matrix(labels_test,labels_pred)


u=pd.DataFrame(labels)
print("Affair percentage is ",round((u[0].value_counts(normalize=True)[1])*100,2))
