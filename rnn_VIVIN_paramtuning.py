# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 17:30:31 2020

@author: vivin
"""
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

def regressor_fn(optm):
    regressor=Sequential()
    #first layer LSTM
    regressor.add(LSTM(units=64,return_sequences=True,input_shape=(X_train.shape[1],1)))
    regressor.add(Dropout(0.2))
    #second layer LSTM
    regressor.add(LSTM(units=64,return_sequences=True))
    regressor.add(Dropout(0.2))#20% dropout
    #third layer LSTM
    regressor.add(LSTM(units=64,return_sequences=True))
    regressor.add(Dropout(0.2))#20% dropout
    #fourth layer
    regressor.add(LSTM(units=64,return_sequences=False))
    regressor.add(Dropout(0.2))#20% dropout
    #output layer
    regressor.add(Dense(units=1))
    #compiling
    regressor.compile(optimizer=optm, loss= 'mean_squared_error')
    return regressor


#parameter tuning
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
regressor = KerasClassifier(build_fn = regressor_fn)
parameters = {'batch_size': [32,50],'epochs': [100,200],
              'optm': ['adam','rmsprop']}
grid_search = GridSearchCV(estimator = regressor, param_grid = parameters,
                           scoring = 'neg_mean_squared_error', cv = 10,
                           return_train_score=True)




#############################
###FITTING RNN TO TRAIN SET##
#############################
grid_search = grid_search.fit(X_train,y_train)

#check which are the best paramters
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

'''
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


#visualising the final results
style.use('dark_background')
plt.title('Google Stock Jan 2017')
plt.xlabel('Time')
plt.ylabel('Open Price')
plt.plot(real_stock_price, label='real',color='red')
plt.plot(predict_stock_price, color='cyan',label='predict')
plt.legend()
plt.show()
regressor.save('Google_stock_2017.h5')
'''


