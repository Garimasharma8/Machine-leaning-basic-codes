#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 17:03:00 2022

@author: garimasharma
"""
# Steps t build CNN are:
# 1. Import librarries and dataset
# 2. Check the data
# 3. perform Exploratory data analysis - visualize, check for missing items, etc
# 4. Perform Data preprocessing - reduce color image to gray to reduce complexity
#    normalize the images (divide by 255)
# 5. Convert Y labels using one hot encoding - vector type
#%% Import fashion MNIST dataset from keras module
import tensorflow as tf
from keras.datasets import fashion_mnist
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split

#%% See the data

(train_X,train_Y), (test_X,test_Y) = fashion_mnist.load_data()

#%% Exploratory data analysis

print("the size of train data is:", train_X.shape)
print("the size of train Y data is", train_Y.shape)

# find the  number of classes in dataset
num_classes = np.unique(train_Y)
print("NUmber of classes are:", len(num_classes))
print(" classes are labeld as:", num_classes)

#%% image visulaization

# display the say 10th image from the training and test dataset

plt.figure(figsize=[28,28])

plt.subplot(121)
plt.imshow(train_X[10,:,:])
plt.title("Actual class : {}".format(train_Y[0]))

plt.subplot(122)
plt.imshow(test_X[10,:,:])
plt.title("Actual class {}".format(test_Y[10]))

#%% Data pre-processing

# reshape images into size 28x28x1
train_X = train_X.reshape(-1,28,28,1)
print(train_X.shape)

test_X = test_X.reshape(-1,28,28,1) 
print(test_X.shape)

# convert images into float32 format from int8

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')

# Map the 0 to 255 level gray scale image into 0 to 1 scale

train_X = train_X/255  # the shape remains same
test_X = test_X/255

#%% convert test classes using one hot encoding 
# keras has a module utils (utilities) and have inbuild function to_catergorical
# to perform one hot encoding

train_y_onehot= tf.keras.utils.to_categorical(train_Y)
test_y_onehot = tf.keras.utils.to_categorical(test_Y)

print("Original class:", train_Y[0])
print("One hot class is:", train_y_onehot[0])

#%% Test train split 80-20 dataset

[train_X, valid_X, train_label, valid_label]= train_test_split(train_X, train_y_onehot, test_size=0.2, shuffle=True, random_state=13)

print("train size is:", train_X.shape)
print("test size is:", valid_X.shape)

#%% Build CNN network, we will use 2 CNN layers - layer 1: 32 filters of size 3 x 3, layer 2: 64 filters of
# size 3 x 3, 2 max pooling layer in between of size 2 x2, then a flattering layer and then a output layer 

# import relevant keras model librarries

from keras.models import Input, Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.activations import leaky_relu
from keras.layers.normalization import batch_normalization
from keras.layers import Conv2D, MaxPool2D
from keras.layers.advanced_activations import LeakyReLU

# initialize batch size, epochs, and number of classes
batch_size = 64 # we can take 128 or 256 or more if memory permits
epochs = 10 # we can increase or decrease
num_classes= 10 
#%%
# select model
model = Sequential()

# %%first layer

model.add(Conv2D(32, kernel_size= (3,3), activation='linear', input_shape = (28,28,1), padding = 'same'))

model.add(LeakyReLU(alpha=0.1))
# max pool layer

model.add(MaxPool2D(pool_size=(2,2), padding='same'))

# Add second layer and max pool layer

model.add(Conv2D(64, kernel_size=(3,3), activation='linear', padding='same'))

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPool2D(pool_size=(2,2), padding='same'))

# Add falttering layer

model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha= 0.1))
# add a dense output layer with 10 nodes (equal to num of classes), use softmax activation

model.add(Dense(num_classes, activation='softmax'))

#%% compile the model with training dataset

# we will use Adam optimizer (better than classical stochastic gradient, Adagrad, RMSprop)
# may check more at: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/

# chose loss = categorical cross entropy as it is a multi class problem

model.compile(optimizer=tf.keras.optimizers.Adam(), loss= tf.keras.losses.categorical_crossentropy, metrics='accuracy')

#%% Check your model 

model.summary()

#%% Fit the model on training data

fashion_train = model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

# after 10 epochs we got accuracy =99%, validation accuracy= 91%. training loss = 0.0245, valid_loss= 0.4672
#looks like model is overfitted, i.e. learnt the training data:
    # solution is to add a dropout layer after each layer,
#%% plot the loss and accuracy of model

test_evaluation = model.evaluate(test_X, test_y_onehot, verbose=0)
print('Test loss:', test_evaluation[0])
print('Test accuracy:', test_evaluation[1])

accuracy = fashion_train.history['accuracy']
val_accuracy = fashion_train.history['val_accuracy']
loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#%% add dropout layers

model_new = Sequential()
model_new.add(Conv2D(32, kernel_size= (3,3), activation='linear', input_shape = (28,28,1), padding = 'same'))

model_new.add(LeakyReLU(alpha=0.1))
# max pool layer

model_new.add(MaxPool2D(pool_size=(2,2), padding='same'))
model_new.add(Dropout(0.1))

model_new.add(Conv2D(64, kernel_size=(3,3), activation='linear', padding='same'))

model_new.add(LeakyReLU(alpha=0.1))

model_new.add(MaxPool2D(pool_size=(2,2), padding='same'))
model_new.add(Dropout(0.1))

model_new.add(Flatten())
model_new.add(Dense(128, activation='linear'))
model_new.add(LeakyReLU(alpha= 0.1))
model_new.add(Dropout(0.1))

model_new.add(Dense(num_classes, activation='softmax'))

#%% 

model_new.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])

model_new.summary()

fashion_train_new = model_new.fit(train_X, train_label, batch_size=64,epochs=10,verbose=1,validation_data=(valid_X, valid_label))

test_evaluation = model_new.evaluate(test_X, test_y_onehot, verbose=0)
print('Test loss:', test_evaluation[0])
print('Test accuracy:', test_evaluation[1])

accuracy = fashion_train_new.history['accuracy']
val_accuracy = fashion_train_new.history['val_accuracy']
loss = fashion_train_new.history['loss']
val_loss = fashion_train_new.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#now if we see, we have a better control on the parameters and model is not overfitted
# we can play with dropout number, it could be 10% or 0.1 or more. 

#%% we can save the trained model to use it later

fashion_train_new.save("fashion_model_dropout.h5py")

#%% Model evaluation test

test_evaluation_new = model_new.evaluate(test_X, test_y_onehot, verbose=1)
print('Test loss:', test_evaluation[0])
print('Test accuracy:', test_evaluation[1])

# after dropout the test loss is reduced even if there is no significant increase in test accuracy

#%% generate classification report

from sklearn.metrics import classification_report

predicted_classes = model_new.predict(test_X)

predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
predicted_classes.shape, test_Y.shape

correct = np.where(predicted_classes==test_Y)[0]
print("Found %d correct labels" ,len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
    plt.tight_layout()
    
incorrect = np.where(predicted_classes!=test_Y)[0]
print("Found %d incorrect labels" ,len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_Y[incorrect]))
    plt.tight_layout()

    

target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_Y, predicted_classes, target_names=target_names))