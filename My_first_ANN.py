#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 14:33:09 2022

@author: garimasharma
"""

#%% first ANN code using keras library and backend tensorflow

# install tensorflow by using commance pip install tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score


#%% import dataset and exploratory data analysis

df= pd.read_csv('/Users/garimasharma/Downloads/Churn_Modelling.csv')
df.head() # view first 5 rows

# look for missing items
df.isnull().sum()

# if no missing item proceed, else either fill missing item or remove missing item rows
X = df.iloc[:, 3:-1].values
y = df.iloc[:, -1].values

print(X)
print(y)


#%% convert categorical data into numerical

# column 2 is male female, import labelencoder for binary labelling

gender = LabelEncoder()
X[:, 2] = gender.fit_transform(X[:, 2])
print(df.head())

# perform one hot encoding for country name
ct = ColumnTransformer(transformers=[ ('encoder', OneHotEncoder(),[1]) ],remainder='passthrough')
X= np.array(ct.fit_transform(X))

#%% now split the data into train test 
[X_train, X_test, y_train, y_test]= train_test_split(X, y, test_size=0.3, random_state=10, shuffle=True)

#%% perform feature scaling to convert features into normal distrubution

sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)  # feature scaling is done on X only which has features

#%% Build ANN model

ann = tf.keras.models.Sequential()  # initialize ANN
#Add input layer and first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))  

#add another hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# add output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#%% train the model

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
 # after compiling fit the model on training set

ann.fit(X_train, y_train, batch_size=30, epochs=50)  

#%% Test our trained model

y_predict=ann.predict(X_test)
y_predict = np.round(y_predict)

#%% calculate performance of our model
 
print("Accuracy rate is:", accuracy_score(y_test, y_predict))
print("Confusion matrix is:", confusion_matrix(y_test, y_predict))
print("precision value is:", precision_score(y_test, y_predict) )
print("recall score is:", recall_score(y_test, y_predict))      
