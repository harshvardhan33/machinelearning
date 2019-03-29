# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:27:20 2019

@author: harshvardhan
"""

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

reg1=LinearRegression()
reg2=LinearRegression()
femaledata=pd.read_csv("stats_females.csv")
femaledata1=pd.read_csv("stats_females.csv")

f_h_median=femaledata['Height'].median()
femaledata1.loc[:,"momheight"]+=1#femaledata1.apply(lambda x: x + 1)


features1=femaledata1.iloc[]
labels1=femaledata1.iloc[]

features2=
labels2=
