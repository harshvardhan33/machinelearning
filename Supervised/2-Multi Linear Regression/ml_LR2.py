# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:14:40 2019

@author: harshvardhan
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


reg=LinearRegression()
lenc=LabelEncoder()
ohe=OneHotEncoder(categorical_features = [0])
dataset=pd.read_csv("Salary_Classification.csv")

temp=dataset.values
features=dataset.iloc[:,:-1].values
labels=dataset.iloc[:,-1:].values


"""This is label encoding which will work on the column 0 values 
it will assign some specifc numerical values to the all unique values
because the machine learning works on the numeric data only """
features[:, 0] = lenc.fit_transform(features[:, 0])

"""now the data in column 1 has been assigned with some kind of numerical 
value but the problem is sometimes the machine learning algorithms 
understands the data in a wrong way considering that there are some kind 
of patterns in the data like 0<1<2, but these values in reality are totaly
independent of any kind of pattern , so to avoid this kind of problem we often
use the OneHotEncoder"""


#features = ohe.fit_transform(features).toarray()
"""IF YOU WANT TO USE THE COLUMN TRANFORMER THEN YOU DONT NEED ANY HELP OF LABEL ENCODER ANYMORE """
transformer = ColumnTransformer(transformers = [("OneHot",OneHotEncoder(),[0])])
features = transformer.fit_transform(features)
features=features[:,1:]


features_train,features_test,labels_train,labels_test=train_test_split(features,labels, test_size = 0.2, random_state = 0)
reg.fit(features_train,labels_train)


reg.predict(features_test)


le1 = lenc.transform(['Development']).reshape(-1,1)
le2 = lenc.transform(['Testing']).reshape(-1,1)
le3 = lenc.transform(['UX']).reshape(-1,1)


dv1 = ohe.transform(le1).toarray()
dv2 = ohe.transform(le2).toarray()
dv3 = ohe.transform(le3).toarray()


"""Pass the data in the same format at which data is trained"""
x = [dv1[0][1], dv1[0][2],3000,2,2]
x = np.array(x).reshape(1,5)






"""
plt.scatter(features_train, labels_train, color = 'red')
plt.plot(features_train, reg.predict(features_train), color = 'blue')
plt.show()
"""



















