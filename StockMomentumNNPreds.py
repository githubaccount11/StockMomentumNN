# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 18:26:03 2022

@author: leigh
"""

import pandas as pd
from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, ModelCheckpoint
import time
from sklearn import preprocessing
import matplotlib.pyplot as plt

import glob
import csv

csv_file_path = 'D:/stocks data/NASDAQpctChange'
    

SEQ_LEN = 30  # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 1  # how far into the future are we trying to predict?
RATIO_TO_PREDICT = "STOCKS"
EPOCHS = 1  # how many passes through our data
BATCH_SIZE = 1  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
NAME = f"{RATIO_TO_PREDICT}-{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

def preprocess_df(df):
        
    df.dropna(inplace=True)
    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), y  # return X and y...and make X a numpy array!



main_df = pd.DataFrame() # begin empty
    
preds=[]
targets=[]
checkpoint = False
for file in glob.glob(csv_file_path + '/*.csv'):
    
    stock = file.replace("D:/stocks data/NASDAQpctChange\\", "")
            
    if stock == "UBCP.csv":
        checkpoint = True
    
    if checkpoint:
        main_df = pd.read_csv(file, names=['date', 'pct_change'])  # read in specific file
    
        main_df.drop('date', axis=1, inplace=True)
        print(file)
        print(main_df)
        main_df.dropna(inplace=True)
        
        main_df['target'] = main_df['pct_change'].shift(-FUTURE_PERIOD_PREDICT)
    
        validation_x, validation_y = preprocess_df(main_df)
    
        if len(validation_x) > 0:
    
            with tf.Graph().as_default():
            
                model = load_model("models/STOCKS-30-SEQ-1-PRED-1643250242-RETURN_SEQUENCES_FALSE")
                preds = model.predict(validation_x)
                
                csv_df = pd.DataFrame()
                
                csv_df['preds'] = preds.tolist()
                csv_df['targets'] = validation_y
                
                csv_df.to_csv(f"D:/stocks data/NASDAQMomentumPreds/{stock}")
    


"""
longestList = 0
for i in range(0, len(preds)):
    if len(preds[i]) > longestList:
        longestList = len(preds[i])
print("LongestList: " + str(longestList))

totalProfit = 0
for i in range(0, longestList - 1):
    bestBet = 0
    for x in range(0, len(preds)):
        if preds[x][i] is not None:
            if preds[x][i] > bestBet:
                bestBet = preds[x][i]
                bestX = x
                bestI = i
    totalProfit = totalProfit + targets[bestX][bestI]

    
    
print("totalProfit: " + str(totalProfit))
    """
    
    
    