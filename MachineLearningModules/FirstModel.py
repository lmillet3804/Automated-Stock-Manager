from alpaca.data.historical import *
from alpaca.data.requests import *
from alpaca.data.timeframe import TimeFrame
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from math import fabs
from datetime import timedelta
import subprocess

API_KEY = "PK03D5OB1PQHPBOUF36Z"
SECRET_KEY = "DhCbFJnwASzcFCwuhjejsKvLg1F6wfO3XJ6k7lZg"

def load_historical_data():
    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    request_params = StockBarsRequest(symbol_or_symbols=["AAPL"], timeframe=TimeFrame.Minute, start="2021-01-01 00:00:00", end="2021-02-08 00:00:00", adjustment=Adjustment.ALL)
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
stock_data['year'] = [d.year for d in stock_data_copy['timestamp']]
stock_data['month'] = [d.month for d in stock_data_copy['timestamp']]
stock_data['day_of_week'] = [d.dayofweek for d in stock_data_copy['timestamp']]
stock_data['hour'] = [d.hour for d in stock_data_copy['timestamp']]
stock_data['minute'] = [d.minute for d in stock_data_copy['timestamp']]
stock_data['timestamp'] = [d for d in stock_data_copy['timestamp']]
stock_data_copy['next_day'] = stock_data_copy['timestamp'] + timedelta(days=1)

load_next_day_price(stock_data, stock_data_copy, pd.Timestamp('2021-02-08 00:00:00').tz_localize('US/Eastern'))

stock_data.drop(['timestamp'], axis=1, inplace=True)


filepath = Path('MachineLearningModules/out.csv')
stock_data.to_csv(filepath)
    
print(stock_data)
stock_data_full = stock_data.copy()

train_set, test_set = train_test_split(stock_data, test_size=0.2, shuffle=False)

stock_data = train_set.copy()