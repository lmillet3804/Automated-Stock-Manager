import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from alpaca.data.historical import *
from alpaca.data.requests import *
from alpaca.data.timeframe import TimeFrame
from datetime import timedelta
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from keras.models import load_model



API_KEY = "PK03D5OB1PQHPBOUF36Z"
SECRET_KEY = "DhCbFJnwASzcFCwuhjejsKvLg1F6wfO3XJ6k7lZg"
START_DATE = "2021-01-01 00:00:00"
FINAL_DATE = "2021-01-31 00:00:00"
WINDOW_SIZE = 20

def load_historical_data():
    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    request_params = StockBarsRequest(symbol_or_symbols=["AAPL"], timeframe=TimeFrame.Minute, start= START_DATE, end=FINAL_DATE, adjustment=Adjustment.ALL)
    return client.get_stock_bars(request_params)

def load_next_day_price(stock_data, stock_data_copy, end_date):
    prices = []
    for row in range(len(stock_data)):
        day_later = stock_data_copy.loc[row].timestamp + timedelta(days=1)
        prices.append(get_next_price(stock_data, stock_data_copy, day_later, end_date, True))
    stock_data['next_day_price'] = prices

def get_next_price(stock_data, stock_data_copy, current_date, end_date, top_level):
    if current_date > end_date:
        return None
    
    if current_date in stock_data.loc['AAPL'].index:
        if top_level:
            return stock_data.loc[('AAPL', current_date)].vwap
        else:
            prev_date = current_date - timedelta(days=1)
            row = stock_data.loc['AAPL'].asof(prev_date)
            if row.timestamp.day == prev_date.day and (row.timestamp.hour > 0 or current_date.hour == 0):
                return None
            else:
                return stock_data.loc[('AAPL', current_date)].vwap
        
    else:
        return get_next_price(stock_data, stock_data_copy, current_date + timedelta(days=1), end_date, False)

stock_data = load_historical_data().df

stock_data_copy = stock_data.copy()

stock_data_copy.reset_index(inplace=True)
# stock_data['year'] = [d.year for d in stock_data_copy['timestamp']]
# stock_data['month'] = [d.month for d in stock_data_copy['timestamp']]
# stock_data['day_of_week'] = [d.dayofweek for d in stock_data_copy['timestamp']]
# stock_data['hour'] = [d.hour for d in stock_data_copy['timestamp']]
# stock_data['minute'] = [d.minute for d in stock_data_copy['timestamp']]
stock_data['timestamp'] = [d for d in stock_data_copy['timestamp']]
stock_data_copy['next_day'] = stock_data_copy['timestamp'] + timedelta(days=1)

load_next_day_price(stock_data, stock_data_copy, pd.Timestamp(FINAL_DATE).tz_localize('US/Eastern'))

stock_data.drop(['timestamp'], axis=1, inplace=True)
stock_data.dropna(subset=["next_day_price"], inplace=True)

df = stock_data.copy()
df.reset_index(inplace=True)
df.index = pd.to_datetime(df['timestamp'])
df.drop(columns=["symbol", "timestamp"], inplace=True)

temp = df['next_day_price']
# temp.plot()
# plt.show()

def df_to_X_y(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [[a] for a in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size]
        y.append(label)
    return np.array(X), np.array(y)

X, y = df_to_X_y(temp, WINDOW_SIZE)

X_train, y_train = X[:10000], y[:10000]
X_val, y_val = X[10000:12000], y[10000:12000]
X_test, y_test = X[12000:], y[12000:]


model1 = Sequential()
model1.add(InputLayer((WINDOW_SIZE, 1)))
model1.add(LSTM(64))
model1.add(Dense(8, 'relu'))
model1.add(Dense(1, 'linear'))

cp = ModelCheckpoint('model1/', save_best_only=True)
model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])

model1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, callbacks=[cp])

model1 = load_model('model1/')

train_predictions = model1.predict(X_train).flatten()
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train})

filepath = Path('MachineLearningModules/out.csv')
train_results.to_csv(filepath)

plt.plot(train_results['Train Predictions'][:10])
plt.plot(train_results['Actuals'][:10])
plt.show()