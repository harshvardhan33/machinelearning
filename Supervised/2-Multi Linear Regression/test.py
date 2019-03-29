# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 11:44:42 2019

@author: harshvardhan
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer



# Importing the dataset
dataset = pd.read_csv('Salary_Classification.csv')
temp = dataset.values
features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, -1].values

# Encoding categorical data

labelencoder = LabelEncoder()
features[:, 0] = labelencoder.fit_transform(features[:, 0])

transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [0]              # The column(s) to be applied on.
         )
    ]
)