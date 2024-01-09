from DataFunctions import download_from_yf, delete_data_ticker, get_df_from_csv, get_data_exist
import os
import shutil
import pytest

def startup():
    folder = 'StockDataTest/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def download_from_yf_test():
    download_from_yf(['AAPL'], filepath='StockDataTest/')
    assert(os.path.exists('StockDataTest/AAPL.csv'))

    download_from_yf(['MSFT', 'TSLA'], filepath='StockDataTest/')
    assert(os.path.exists('StockDataTest/MSFT.csv'))
    assert(os.path.exists('StockDataTest/TSLA.csv'))

def get_data_exist_test():
    download_from_yf(['AAPL'], filepath='StockDataTest/')
    assert(get_data_exist('AAPL', filepath='StockDataTest/'))
    assert(not get_data_exist('TSLA', filepath='StockDataTest/'))

def delete_data_ticker_test():
    download_from_yf(['AAPL'], filepath='StockDataTest/')
    delete_data_ticker('AAPL', filepath='StockDataTest/')
    assert(not os.path.exists('StockDataTest/AAPL.csv'))
    delete_data_ticker('AAPL', filepath='StockDataTest/')
    assert(not os.path.exists('StockDataTest/AAPL.csv'))

def get_df_from_csv_test():
    download_from_yf(['AAPL', 'MSFT'], filepath='StockDataTest/')
    aapl_df = get_df_from_csv('AAPL', filepath='StockDataTest/')
    msft_df = get_df_from_csv('MSFT', filepath='StockDataTest/')
    tsla_df = get_df_from_csv('TSLA', filepath='StockDataTest/')
    assert(aapl_df.columns.all() == msft_df.columns.all())
    assert(msft_df.columns.all() == tsla_df.columns.all())


startup()
get_df_from_csv_test()