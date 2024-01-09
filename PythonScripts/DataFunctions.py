import yfinance as yf
import pandas as pd
import os
from pathlib import Path

def download_from_yf(tickers:list, start_date:str=None, end_date:str=None, filepath:str='StockData/') -> None:
    for ticker in tickers:
        req = yf.download(ticker, start_date, end_date)
        path = Path(f'{filepath}{ticker}.csv')
        req.to_csv(path)

def get_data_exist(ticker:str, filepath:str='StockData/') -> bool:
    return os.path.exists(f'{filepath}{ticker}.csv')

def delete_data_ticker(ticker:str, filepath:str='StockData/'):
    if get_data_exist(ticker, filepath):
        os.remove(f'{filepath}{ticker}.csv')
    else:
        print(f'{ticker} was not found in {filepath}')

def get_df_from_csv(ticker:str, filepath:str='StockData/') -> pd.DataFrame:
    if not get_data_exist(ticker):
        download_from_yf(tickers=[ticker], filepath=filepath)
    df = pd.read_csv(f'{filepath}{ticker}.csv')
    return df