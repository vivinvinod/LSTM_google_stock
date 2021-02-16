# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:06:12 2020

@author: vivin
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd


#Importing the training set
df_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = df_train.iloc[:,1:2].values #to create numpy array

####PREPROCESSING_DATA##########
#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
scaled_training_set = sc.fit_transform(training_set) #normalised to b/w 0 and 1

#creating datastructure
X_train = []
y_train = []
for i in range(60,1258):
    X_train.append(scaled_training_set[i-60:i,0])
    y_train.append(scaled_training_set[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

#reshaping the data
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

#Building the RNN
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

##################################
########RNN ARCHITECTURE##########
##################################
#initialising RNN
regressor = Sequential()

#Adding first LSTM and dropout regularisation
regressor.add(LSTM(units=64,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))#20% dropout

#adding 2nd LSTM and dropout
regressor.add(LSTM(units=64,return_sequences=True))
regressor.add(Dropout(0.2))#20% dropout

#adding 3rd LSTM and dropout
regressor.add(LSTM(units=64,return_sequences=True))
regressor.add(Dropout(0.2))#20% dropout

#adding 4th LSTM and dropout
regressor.add(LSTM(units=64,return_sequences=False))
regressor.add(Dropout(0.2))#20% dropout

#Adding output layer
regressor.add(Dense(units=1))

##############################
#Compiling the RNN
regressor.compile(optimizer='adam',loss='mean_squared_error')


###FITTING RNN TO TRAIN SET##
#############################
regressor.fit(X_train,y_train,epochs=100,batch_size=32)

#save the model for later use/training whatever
regressor.save('Google_stock_2017.h5')


#making predictions

#get the stock price - real
df_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = df_test.iloc[:,1:2].values

#getting predicted test
df_total = pd.concat((df_train['Open'], df_test['Open']),axis=0)
inputs = df_total[len(df_total)-len(df_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])

X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1], 1)) #turning it into 3D

predict_stock_price = regressor.predict(X_test)

#inverting the scaling
predict_stock_price = sc.inverse_transform(predict_stock_price)

diff = predict_stock_price - real_stock_price
print('Mean of difference:',
      np.mean(diff)/(np.max(real_stock_price)-np.min(real_stock_price)))
print('Standard Deviation of difference:',
      np.std(diff)/(np.max(real_stock_price)-np.min(real_stock_price))) 
#division by range of real_stock_price for realtive error and not absolute error

#visualising the final results
style.use('dark_background')
plt.figure(0)
plt.title('Google Stock Jan 2017')
plt.xlabel('Time')
plt.ylabel('Open Price')
plt.plot(real_stock_price, label='real',color='red')
plt.plot(predict_stock_price, color='cyan',label='predict')
plt.legend()

#Errors
f,p_arr = plt.subplots(2,sharex=True)
f.suptitle('Errors in Prediction')
p_arr[0].plot(diff/(np.max(real_stock_price)-np.min(real_stock_price)),
         label='relative error')
p_arr[0].plot(diff,label='absolute error',color='red')

p_arr[0].legend()
p_arr[1].plot(diff/(np.max(real_stock_price)-np.min(real_stock_price)),
         label='relative error')
plt.show()



