# Automated Stock Manager
Stock Manager using AI

This project aims to introduce me to machine learning and neural networks. Using past days' stock prices, this project ultimately outputs a corresponding value to each stock representing the percentage of money to invest in that stock in order to produce a maximum return.

## Mechanics
The program first uses the *yfinance* library to fetch data from the past with given stock tickers. This data is preprocessed and sent to the first neural network: an LSTM model. This model outputs predicted returns for the next 24 hours for each stock ticker. Each prediction is compiled into a master list, which consists of all predicted returns for each stock ticker. This list is then sent to the second neural network: a custom model built from scratch, which outputs a decimal between 0 and 1 for each stock representing what percentage of a portfolio should be invested in the respective stock for a maximum profit. This model ideally should handle stock volatility, the accuracy of the previous model, as well as anything more that might influence a reason to buy or sell a stock.

### LSTM Model
This model was built using the *tensorflow* and *keras* python libraries. It preprocesses the input data by calculating the percent increase from the current day to the following day for each entry in the data. Then using this daily return percentage (or drp), for every entry, it compiles the last three days of drp as input (or X). The label (or y) is simply the drp for that given day. Therefore, the input shape given to the network is (3, 1) and it returns a single value.

The model itself is built from four layers (excluding the input and output layers). The first two are LSTM layers with 64 and 32 nodes respectively, and the last two are Dense layers with ReLU activitation functions and 16 nodes each. It is trained on several years worth of data, uses *MeanSquaredError* loss and the *Adam* optimizer. 

### Custom Neural Network
This model was built entirely from the ground up with no external libraries. It works by taking input into a layer, for each node passing it through an activation function, then sending the output to the next layer while factoring in an extra weight, until it gets to the output layer. If the sum of the output percentages is greater than 1, they are sent through a softmax function. The training works with a custom loss function, which is computed with the following: <br />
R = O • Y <br />
L = e<sup>-R </sup> <br />
where O is the network output vector, Y is the actual return vector, R is the predicted portfolio return, and L is the loss. Starting from the output layer, a recursive function takes the partial derivative of each node's bias and each weight's connection with respect to the loss function and adds it to a gradient vector. Each weight and bias is then altered by its corresponding value in the gradient vector.

The model itself is built with 2 layers (excluding the input and output layers). Both interior layers have a ReLU activiation function and 16 nodes, while the input and output layers are dependent on how many tickers to track.

## Problems
The most glaring problem in this problem is the model is very overfit to the training data. Running a simulation with 10 epochs gave a 500% return over about 9 years of training data, while the validation data gave a return of about -50% in a little over a year. I think this is a result of the input data from the first model. Altering and fixing the LSTM model to more accurately predict stock returns will better the second neural network because of better data.

Another problem is the network does not support many stocks to track. Tracking more than a few stocks gives an overflow error when calculating the denominator of the softmax function. This means the solution to what should happen when the percentage of the portfolio exceeds 100% should be rethought. 

## Goals
- [ ] Create a node.js script and web front-end to continuously run and interact with the program to create a truly *automated* stock manager
- [ ] Integrate with Alpaca's API to run a paper trading account and test the algorithm
- [ ] Optimize the training process to run faster so it can run more often
