# Automated Stock Manager
Stock Manager using AI

This project aims to introduce me to machine learning and neural networks. Using past days' stock prices, this project ultimately outputs a corresponding value to each stock representing the percentage of money to invest in that stock in order to produce a maximum return.

## Mechanics
The program first uses the *yfinance* package to fetch data from the past with given stock tickers. This data is preprocessed and sent to the first neural network: an LSTM model. This model outputs predicted returns for the next 24 hours for each stock ticker. Each prediction is compiled into a master list, which consists of all predicted returns for each stock ticker. This list is then sent to the second neural network: a custom model built from scratch, which outputs a decimal between 0 and 1 for each stock representing what percentage of a portfolio should be invested in the respective stock for a maximum profit. This model ideally should handle stock volatility, the accuracy of the previous model, as well as anything more that might influence a reason to buy or sell a stock.

### LSTM Model
This model was built using the *tensorflow* and *keras* python packages. It preprocesses the input data by calculating the percent increase from the current day to the following day for each entry in the data. Then using this daily return percentage (or drp), for every entry, it compiles the last three days of drp as input (or X). The label (or y) is simply the drp for that given day. Therefore, the input shape given to the network is (3, 1) and it returns a single value.

The model itself is built from four layers (excluding the input and output layers). The first two are LSTM layers with 64 and 32 nodes respectively, and the last two are Dense layers with ReLU activitation functions and 16 nodes each. It is trained on several years worth of data, uses *MeanSquaredError* loss and the *Adam* optimizer. 
