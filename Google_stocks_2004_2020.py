# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 22:44:56 2020

@author: vivin
"""
'''
import numpy as np
import matplotlib.pyplot as plt #for plotting (final result)
import pandas as pd #for data structures
from sklearn.preprocessing import MinMaxScaler #feature scaling
from keras.models import Sequential #model architecture
from keras.layers import Dense, LSTM, Dropout #model architecture
import os
os.chdir(r"D:\Work\Udemy\DeepLEarningA-Z\Resource files\Recurrent_Neural_Networks")
#importing the main dataset
training_dataset = pd.read_csv('GOOG.csv')
training_set = training_dataset.iloc[:,1:2].values #to create numpy array

####PREPROCESSING_DATA##########
#Feature Scaling
sc = MinMaxScaler(feature_range=(0,1)) #noramlisation
scaled_training_set = sc.fit_transform(training_set) #normalised to b/w 0 and 1

#creating datastructure
n = int(100) #time steps to consider prior tp prediction (days in this case)
X_train = []
y_train = []
for i in range(n,3930):
    X_train.append(scaled_training_set[i-n:i,0])
    y_train.append(scaled_training_set[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

#reshaping the data
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

#Building the RNN
########RNN ARCHITECTURE##########

#initialising RNN
regressor = Sequential()

#Adding first LSTM and dropout regularisation
regressor.add(LSTM(units=256,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.3))#30% dropout

#adding 2nd LSTM and dropout
regressor.add(LSTM(units=256,return_sequences=True))
regressor.add(Dropout(0.3))#30% dropout

#adding 3rd LSTM and dropout
regressor.add(LSTM(units=256,return_sequences=True))
regressor.add(Dropout(0.3))#30% dropout

#adding 4th LSTM and dropout
regressor.add(LSTM(units=256,return_sequences=False))
regressor.add(Dropout(0.3))#30% dropout

#Adding output layer
regressor.add(Dense(units=1))


#Compiling the RNN
regressor.compile(optimizer='adam',loss='mean_squared_error')

#############################
###FITTING RNN TO TRAIN SET##
#############################
regressor.fit(X_train,y_train,epochs=200,batch_size=32)
'''

#Saving regressor?
#regressor.save('Google_stock_daily.h5')
############################
X_test = scaled_training_set[0:100,1:2]
X_test=np.reshape(X_test,(X_test.shape[1],X_test.shape[0],1))
