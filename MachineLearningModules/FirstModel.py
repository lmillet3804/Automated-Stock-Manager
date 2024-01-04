from alpaca.data.historical import *
from alpaca.data.requests import *
from alpaca.data.timeframe import TimeFrame
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
from math import fabs
from datetime import timedelta
import subprocess
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

API_KEY = "PK03D5OB1PQHPBOUF36Z"
SECRET_KEY = "DhCbFJnwASzcFCwuhjejsKvLg1F6wfO3XJ6k7lZg"
START_DATE = "2021-01-01 00:00:00"
FINAL_DATE = "2021-01-31 00:00:00"

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
stock_data['year'] = [d.year for d in stock_data_copy['timestamp']]
stock_data['month'] = [d.month for d in stock_data_copy['timestamp']]
stock_data['day_of_week'] = [d.dayofweek for d in stock_data_copy['timestamp']]
stock_data['hour'] = [d.hour for d in stock_data_copy['timestamp']]
stock_data['minute'] = [d.minute for d in stock_data_copy['timestamp']]
stock_data['timestamp'] = [d for d in stock_data_copy['timestamp']]
stock_data_copy['next_day'] = stock_data_copy['timestamp'] + timedelta(days=1)

load_next_day_price(stock_data, stock_data_copy, pd.Timestamp(FINAL_DATE).tz_localize('US/Eastern'))

stock_data.drop(['timestamp'], axis=1, inplace=True)
stock_data.dropna(subset=["next_day_price"], inplace=True)


#filepath = Path('MachineLearningModules/out.csv')
#stock_data.to_csv(filepath)
    
# print(stock_data)
# print(stock_data.info())
# print(stock_data.describe())

stock_data_full = stock_data.copy()

train_set, test_set = train_test_split(stock_data, test_size=0.2, shuffle=False)

print(train_set.tail())
stock_data = train_set.copy()
stock_data_labels = stock_data["next_day_price"]

stock_data.reset_index(inplace=True)
stock_data.drop(columns=["next_day_price", "symbol", "timestamp", "year", "day_of_week", "hour", "minute", "month", "volume", "trade_count"], inplace=True)
#print(stock_data.head())

std_scaler = StandardScaler()
stock_data_std_scaled = std_scaler.fit_transform(stock_data)

lin_reg = LinearRegression()
lin_reg.fit(stock_data_std_scaled, stock_data_labels)

test_labels = test_set["next_day_price"]
original_values = test_labels

test_set.reset_index(inplace=True)
test_set.drop(columns=["next_day_price", "symbol", "timestamp", "year", "day_of_week", "hour", "minute", "month", "volume", "trade_count"], inplace=True)

print(test_set.head())

#print(test_labels)
test_data_std_scaled = std_scaler.fit_transform(test_set)

print(original_values)
predicted_values = lin_reg.predict(test_data_std_scaled)


print(predicted_values)

comparison_dataframe = pd.DataFrame(data={"Original Values": original_values, "Predicted Values":predicted_values})
comparison_dataframe["Difference %"] = (comparison_dataframe["Original Values"] - comparison_dataframe["Predicted Values"]) / comparison_dataframe["Original Values"] * 100

filepath = Path('MachineLearningModules/predictions.csv')
comparison_dataframe.to_csv(filepath)