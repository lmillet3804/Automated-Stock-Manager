import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import InputLayer
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.models import load_model
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import math
import DataFunctions
import os

def sig_scale(input):
    return 1 / (1 + math.exp(-25 * input))

def df_to_X_y_dates(df, n=3, start_date='2013-04-29', end_date='2024-01-05'):
    X=[]
    y=[]
    dates = []

    df = df[start_date:end_date]
    df = df.reset_index()
    for i in range(n, len(df)):
        x = []
        for j in range(i-n, i):
            newx = []
            newx.append(df.iloc[j]['drp_scaled'])
            #newx.append(df.iloc[j]['Volume'])
            x.append(newx)
        X.append(x)
        y.append(df.iloc[i]['drp_scaled'])
        dates.append(df.iloc[i]['Date'])
    print(X[:5])
    return X, y, dates

def add_drp_col(df):
    new_row = [0]
    for i in range(1,len(df['Adj Close'])):
        new_row.append((df.iloc[i]['Adj Close'] - df.iloc[i-1]['Adj Close']) / df.iloc[i-1]['Adj Close'])
    df['drp'] = new_row

    df['drp_scaled'] = df['drp'].apply(sig_scale)
    return df

def train_model_for_ticker(ticker:str):
    df = DataFunctions.get_df_from_csv(ticker)
    dates = pd.to_datetime(df['Date'])

    df = add_drp_col(df)
    print(df.describe())
    df.index = df.pop('Date')
    X, y, dates = df_to_X_y_dates(df)
    real_drp = df['drp']
    df.reset_index(inplace=True)

    q_80 = int(len(y) * 0.8)
    q_90 = int(len(y) * 0.9)

    dates_train, dates_val, dates_test = dates[:q_80], dates[q_80:q_90], dates[q_90:]
    X_train, X_val, X_test = X[:q_80], X[q_80:q_90], X[q_90:]
    y_train, y_val, y_test = y[:q_80], y[q_80:q_90], y[q_90:]
    real_drp_train, real_drp_val, real_drp_test = real_drp[:q_80], real_drp[q_80:q_90], real_drp[q_90]

    model = Sequential()
    model.add(InputLayer((3, 1)))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(16, 'relu'))
    model.add(Dense(16, 'relu'))
    model.add(Dense(1, 'linear'))

    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001))
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
    model.save(f'MachineLearningModules/{ticker}_Model')

    train_predictions = model.predict(X_train).flatten()
    val_predictions = model.predict(X_val).flatten()
    test_predictions = model.predict(X_test).flatten()

    return train_predictions, val_predictions, test_predictions, dates_train, dates_val, dates_test, real_drp_train, real_drp_val, real_drp_test

def load_model_for_ticker(ticker:str):
    if os.path.exists(f'MachineLearningModules/{ticker}_Model'):
        model = load_model(f'MachineLearningModules/{ticker}_Model')
        df = DataFunctions.get_df_from_csv(ticker)
        dates = pd.to_datetime(df['Date'])

        df = add_drp_col(df)
        print(df.describe())
        df.index = df.pop('Date')
        X, y, dates = df_to_X_y_dates(df)
        real_drp = df['drp']
        df.reset_index(inplace=True)

        q_80 = int(len(y) * 0.8)
        q_90 = int(len(y) * 0.9)

        dates_train, dates_val, dates_test = dates[:q_80], dates[q_80:q_90], dates[q_90:]
        X_train, X_val, X_test = X[:q_80], X[q_80:q_90], X[q_90:]
        y_train, y_val, y_test = y[:q_80], y[q_80:q_90], y[q_90:]
        real_drp_train, real_drp_val, real_drp_test = real_drp[:q_80], real_drp[q_80:q_90], real_drp[q_90]

        train_predictions = model.predict(X_train).flatten()
        val_predictions = model.predict(X_val).flatten()
        test_predictions = model.predict(X_test).flatten()

        return train_predictions, val_predictions, test_predictions, dates_train, dates_val, dates_test, real_drp_train, real_drp_val, real_drp_test
    else:
        return train_model_for_ticker(ticker)