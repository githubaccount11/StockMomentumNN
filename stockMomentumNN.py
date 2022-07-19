# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 14:48:37 2021

@author: leigh
"""

import pandas as pd
import math
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
NAME = f"{RATIO_TO_PREDICT}-{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}-RETURN_SEQUENCES_FALSE"

def preprocess_df(df):
    df.dropna(inplace=True)
    print("preprocessing:")
    print(df)
    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    cap = 0
    for i in df.values:  # iterate over the values(columns)
        prev_days.append([n for n in i[:-1]])  # store all (rows) but the last target
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!
            if cap < 10:
                print(np.array(prev_days))
                print(i[-1])
                cap = cap + 1
    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), y  # return X and y...and make X a numpy array!



main_df = pd.DataFrame() # begin empty

for file in glob.glob(csv_file_path + '/*.csv'):
    
    df = pd.read_csv(file, names=['date', 'pct_change'])  # read in specific file
    
    df.drop('date', axis=1, inplace=True)

    if len(main_df)==0:  # if the dataframe is empty
        main_df = df  # then it's just the current df
    else:  # otherwise, join this data to the main one
        main_df = main_df.append(df)
            
        
print(main_df)
main_df.dropna(inplace=True)
main_df['target'] = main_df['pct_change'].shift(-FUTURE_PERIOD_PREDICT)

times = sorted(main_df.index.values)
last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))]


validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)


print(f"train data: {len(train_x)} validation: {len(validation_x)}")


model = Sequential()
model.add(LSTM(30, input_shape=(train_x.shape[1:]), return_sequences=False))
model.add(BatchNormalization())

model.add(Dense(1, activation='tanh'))


opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6)

# Compile model
model.compile(
    loss='mean_squared_error',
    optimizer=opt,
)

# Train model
history = model.fit(
    np.array(train_x), np.array(train_y),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(np.array(validation_x), np.array(validation_y)),
)

# Save model
model.save("models\\{}".format(NAME))