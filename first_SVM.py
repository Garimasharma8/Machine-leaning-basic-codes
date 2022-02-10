#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 13:53:02 2022

@author: garimasharma
"""

#%% code to try SVM on python

# import libraries
import pandas as pd
from sklearn import datasets
import matplotlib
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
#%% import dataset

cancer = datasets.load_breast_cancer()

#%% exploratory data analysis

print("Features:", cancer.feature_names)

print("lables", cancer.target_names)

cancer.data.shape

# print first 5 rows (use head or indexing)
print(cancer.data[0:5])

# print target lables
print(cancer.target)

#%% test train split

[X_train, X_test, y_train, y_test] = train_test_split(cancer.data,cancer.target, test_size=0.3, random_state=10, shuffle=True)

#%% Generate SVM model

model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)  #fit() means train the model

y_predict = model.predict(X_test)

#%% evaluate model performance via metric


print("Accuracy rate:", metrics.accuracy_score(y_test, y_predict)*100) #accuracy in %

print("Confusion matrix is:", metrics.confusion_matrix(y_test, y_predict))

print("Model's precision is: ", metrics.precision_score(y_test, y_predict))
print("model's recall is: ", metrics.recall_score(y_test, y_predict))

