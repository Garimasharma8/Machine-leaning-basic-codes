#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 10:04:39 2022

@author: garimasharma
"""

# my first linear regression algorithm on churn modelling dataset
#%% Import libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import seaborn as sns


#%% import churn modelling dataset using pandas

df = pd.read_csv('/Users/garimasharma/Downloads/Churn_Modelling.csv')

#%% exploratory data analysis

print(df.head()) # check first 5 rows
missing_data= df.isnull().sum()  # check for missing data

# we will work only with numeric columns here and delete categorical data columns, which are column 2,4,5

del df['Surname']
del df['Geography']
del df['Gender']

# check the heatmap to see correlarions between independent variables

df_heat = df.corr()
sns.heatmap(df_heat, annot = True, cmap='coolwarm')

# we can see there are no correlated independent variables, otherwise we would be deleting the highly correlated 
# independent variables 

#%% divide dataset into train test sets

[X_train, X_test, y_train, y_test]= train_test_split(df.iloc[:,0:10], df.iloc[:,-1], test_size=0.3, shuffle=(True))

#%% fit the model

LR_model = LinearRegression()

LR_model.fit(X_train,y_train)

#%% predict the model performance on test set

y_pred = LR_model.predict(X_test)

#%% check performance metrics

print("Mean absolute error is:", mean_absolute_error(y_test,y_pred))
