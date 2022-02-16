# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 15:12:18 2022

@author: g26sharm
"""

# to predict up-ward or down-ward trend in google stock price , dataset train and test is available
# the main steps to build RNN are: 
# 1. Import libraries 
# 2. Import dataset
# 3. Feature scaling
# 4. select timesteps and build training and label data
# 5. reshape or add new dimensions if needed
# 6. Build RNN architecture
# 7. Compile RNN
# 8. Fit data on RNN
# 9. Predict on test data
# 10. Visulaize the results
    

#%% Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score
from keras.layers import LSTM, Dense, Dropout

#%% Import training set

dataset_train = pd.read_csv('Train.csv')
# actual training set on which lstm will be trained

training_set = dataset_train.iloc[:,1:2].values  #numpy array of open price only

#%% Feature scaling
# two best ways of feature scaling are: standardization and normalization

# normalization is good option when sigmoid is in use

sc = MinMaxScaler(feature_range=(0,1))  # convert all features in range 0 to 1
# apply SC to our data

scaled_training_set = sc.fit_transform(training_set)

#%% Number of time steps RNN has to remember
# create a data structure with 60 timesteps and 1 output (account 60 timesteps and understand some trends
# and predict 1 output, optimum no of time steps could be calulated imperically)

X_train=[] 
Y_train=[]

for i in range(60,1258):
    X_train.append(scaled_training_set[i-60:i,0])
    Y_train.append(scaled_training_set[i,0])

X_train, Y_train = np.array(X_train), np.array( Y_train)  # make x train and y train as numpy array so that it could be input to RNN    

#%% Reshaping / add new dimentionor predictor, always use reshape when we want to add new dimension to numpy array
# new dimension could be something that could effect the decision e.g. closed stock price etc. 

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  #refer keras documentation 

#%% Build RNN - architecture

# Initialize RNN

model = Sequential()

# add layers

model.add(LSTM(units=50, activation='tanh', dropout=0.2, return_sequences=True, input_shape = ( X_train.shape[1], 1)))

# Add second LSTM layer with some dropout regularization
model.add(LSTM(units=50, activation='tanh', return_sequences=True, dropout=0.2))

# add third and fourth layer
model.add(LSTM(units=50, activation='tanh', return_sequences=True, dropout=0.2))
# make return sequences false as we don't want another layer
model.add(LSTM(units=50, activation='tanh', return_sequences=False, dropout=0.2)) 

# add output final layer
model.add(Dense(units=1))

#%% compile RNN, then fit it on training set, then prediction

model.compile(optimizer='adam', loss='mean_squared_error')     #rmsprop is a good choice for RNN, 
#refer keras documentaion, you can use adam too, loss can't be binary as this is not a classification problem, hence we go for
# MSE

#%% fit RNN to training set

model.fit(X_train, Y_train, epochs=100, batch_size=50)  # choose epochs imperically from anywhere between 25 to 100 or more. 
#check for convergance 

#%% Predict on test data

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')

real_stock_price = dataset_test.iloc[:,1:2].values
#%% predict the stock price for Jan 2017

# concatenate the training and test set as to predict jan 2017 stock price, we also need price of jan month

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0) #axis is 0 as we want to concatenate rows

# get the stock prices of 3 previous months or 60 days (exclude sat and sundays)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values # it has all inputs for 
#predicting jan stock prices


inputs = inputs.reshape(-1,1)   # one column form

#scale the inputs too
scaled_inputs = sc.transform(inputs)

## create 3d shape for input 
X_test=[]

for i in range(60, 80):
    X_test.append(inputs[i-60:i,0])
    
X_test = np.array(X_test) 

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = model.predict(X_test)

#%%

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#%% inverse the scaling of inputs/train and test

predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#%% visulaizing the final results

import matplotlib.pyplot as plt
import seaborn as sns

plt.plot(real_stock_price, color='red', label='real google stock price')
plt.plot(predicted_stock_price, color='blue', label='predicted google stock price')
plt.title('Google stock price prediction')
plt.xlabel('Time')
plt.ylabel('google stock price')
plt.legend()
plt.show()





    
