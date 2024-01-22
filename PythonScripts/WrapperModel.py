from StockTrainingModel import train_model_for_ticker
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import keras.backend as K
import math

class Node:
    pass

class Layer:
    pass

class InputLayer(Layer):
    pass

class OutputLayer(Layer):
    pass

class Connection:
    pass

class NeuralNetwork:
    pass

stocks_to_trade = ['MSFT', 'AAPL']
training_data_df = pd.DataFrame()
actual_train_df = pd.DataFrame()
val_data_df = pd.DataFrame()
actual_val_df = pd.DataFrame()
test_data_df = pd.DataFrame()
actual_test_df = pd.DataFrame()

for stock in stocks_to_trade:
    new_train, new_val, new_test, new_train_dates, new_val_dates, new_test_dates, real_train, real_val, real_test = train_model_for_ticker(stock)
    training_data_df.insert(stocks_to_trade.index(stock), stock, new_train)
    actual_train_df.insert(stocks_to_trade.index(stock), f'{stock}_real', real_train)
    val_data_df.insert(stocks_to_trade.index(stock), stock, new_val)
    actual_val_df.insert(stocks_to_trade.index(stock), f'{stock}_real', real_val)
    test_data_df.insert(stocks_to_trade.index(stock), stock, new_test)
    actual_test_df.insert(stocks_to_trade.index(stock), f'{stock}_real', real_test)

input_layer = Input((len(stocks_to_trade),))
dense_1 = Dense(128)(input_layer)

output_1 = Dense(1, activation='relu')(dense_1)
output_2 = Dense(1, activation='relu')(dense_1)

label_layer_1 = Input((1,))
label_layer_2 = Input((1,))

model = Model(inputs=[input_layer, label_layer_1, label_layer_2], outputs=[output_1, output_2])
# model.add(Dense(12, input_shape=(len(stocks_to_trade),), activation='sigmoid'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(len(stocks_to_trade), activation='linear'))

loss = 1 / (output_1 * label_layer_1 + output_2 * label_layer_2)
def loss_scaled (out_1, out_2, label_1, label_2):
    new_out_1 = out_1 / out_1 + out_2
    new_out_2 = out_2 / out_1 + out_2
    out_1 = new_out_1
    out_2 = new_out_2
    return loss_unscaled(out_1, out_2, label_1, label_2)

def loss_unscaled(out_1, out_2, label_1, label_2):
    r = label_1 * out_1 + label_2 * out_2
    #print(K.eval(model.layers[2].output))
    return math.exp(1) ** (-1*r)

def custom_loss_function(out_1, out_2, label_1, label_2):
    #out_1 = max(out_1, 0)
    #out_2 = max(out_2, 0)
    print(out_1)
    #print(K.get_value(out_1))
    print(out_2)
    # if out_1 + out_2 > 1:
    #     new_out_1 = out_1 / out_1 + out_2
    #     new_out_2 = out_2 / out_1 + out_2
    #     out_1 = new_out_1
    #     out_2 = new_out_2
    # #sum = 0
    # r = label_1 * out_1 + label_2 * out_2
    # return 1 / r
    #return 1 / (out_1 * label_1 + out_2 * label_2)
    return K.switch(out_1 + out_2 > 1, loss_scaled(out_1, out_2, label_1, label_2), loss_unscaled(out_1, out_2, label_1, label_2))
    # if out_1 > out_2:
    #     return out_1 + out_2
    # return out_1 - out_2
loss = custom_loss_function(output_1, output_2, label_layer_1, label_layer_2)
model.add_loss(loss)

dummy_train = np.zeros((len(actual_train_df),))
dummy_val = np.zeros((len(actual_val_df),))
model.compile(optimizer=Adam(learning_rate=-0.01))
model.fit([training_data_df, actual_train_df['MSFT_real'], actual_train_df['AAPL_real']], dummy_train, validation_data=([val_data_df, actual_val_df['MSFT_real'], actual_val_df['AAPL_real']], dummy_val), epochs=10)

train_predictions = model.predict([training_data_df, actual_train_df['MSFT_real'], actual_train_df['AAPL_real']])
print(len(train_predictions[0][0]))