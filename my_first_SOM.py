 # -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 12:29:49 2022

@author: garima Sharma

"""
# Self -organizing maps- Unsupervised ML algo, used for feature reduction
# generates a colourful map that reflects different clusters based on various features/predictors
#%% Import Libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#%% Install SOM

# pip install MiniSom

# run this command in terminal

#%%  Import dataset

df = pd.read_csv(r'C:\Users\g26sharm\Desktop\P16-Self-Organizing-Maps\Self_Organizing_Maps\Credit_Card_Applications.csv')
x = df.iloc[:,:-1].values  # select all columns except last
y = df.iloc[:,-1].values       # select  last column only

#%% Feature scaling

sc = MinMaxScaler(feature_range=(0,1))

x_scaled = sc.fit_transform(x)

#%% Train Self organizing maps

# take SOM from other module or developer we will use- MInisom 1.0 , required numpy
from minisom import MiniSom  # we imported class minisom from python file minison that is in working dir

model_som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5) 
# x = 10, y=10 so that we have 10 by 10 grid

# initialize the weights
model_som.random_weights_init(x_scaled)

model_som.train_random(data = x_scaled, num_iteration=100)

#%% visualize the results i.e. plot SOM

from pylab import bone, pcolor, colorbar, plot, show

bone()  # will create a white figure, initialize a figure by bone

pcolor(model_som.distance_map().T)  # shows MIDs
colorbar()  # highest MOD is shown by white color, hence white nodes reflects frauds or outliers

#red circles, customers who didnt get approval
markers = ['o', 's' ] # circle and square
colors = ['r','g' ] # red and green color

for i, x in enumerate(x_scaled):
    w = model_som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()
#%% inverse mapping of winning node to customer ID's

mappings = model_som.win_map(x_scaled)
frauds = np.concatenate((mappings[(6,8)], mappings[(5,1)]), axis = 0)
frauds = sc.inverse_transform(frauds)