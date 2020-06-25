# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:04:41 2020

@author: vivin
"""

import pandas as pd
from keras.models import load_model
import os
os.chdir(r"D:\Work\Udemy\DeepLEarningA-Z\Resource files\Recurrent_Neural_Networks")

#Data
df_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = df_train.iloc[:,1:2].values #to create numpy array


#load model
model = load_model('Google_stock_daily.h5')
model.summary() #model summary

#dataset
