from StockTrainingModel import train_model_for_ticker
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from enum import Enum
import pandas as pd
import numpy as np
import tensorflow as tf
import keras.backend as K
import math
import random

class Activation_Functions(Enum):
    IDENTITY = 0
    RELU = 1
    SIGMOID = 2
    SOFTPLUS = 3

class Node:
    #The fundamental building block of the network
    #It has input, activation function, and output
    def __init__(self, activation:Activation_Functions = Activation_Functions.RELU) -> None:
        self.pred_value = 0
        self.input_value = 0
        self.output_value = 0
        self.activation_str = activation
        self.bias = 0
        self.in_connections = []
        self.out_connections = []
        self.network_index = -1

    def evaluate(self):
        if not (self.network_index == -1):            
            sum = 0
            for connection in self.in_connections:
                sum += connection.value * connection.weight
            self.pred_value = sum
            self.input_value = sum + self.bias

        if self.activation_str == Activation_Functions.IDENTITY:
            self.output_value = self.input_value
        elif self.activation_str == Activation_Functions.RELU:
            self.output_value = max(self.input_value, 0)
        elif self.activation_str == Activation_Functions.SIGMOID:
            self.output_value = 1 / (1 + math.exp(-1 * self.input_value))
        elif self.activation_str == Activation_Functions.SOFTPLUS:
            self.output_value = math.exp(self.input_value) / (1 + math.exp(self.input_value))
        
        for connection in self.out_connections:
            connection.value = self.output_value

    def __node_grad(self, connection):
        if self.activation_str == Activation_Functions.IDENTITY:
            return connection.weight
        elif self.activation_str == Activation_Functions.RELU:
            if self.input_value > 0:
                return connection.weight
            else:
                return 0
        elif self.activation_str == Activation_Functions.SIGMOID:
            return (math.exp(-1 * self.input_value) * connection.weight) / (1 + math.exp(-1 * self.input_value))
        
    def __weight_grad(self, connection):
        if self.activation_str == Activation_Functions.IDENTITY:
            return connection.prev_node.output_value
        elif self.activation_str == Activation_Functions.RELU:
            if self.input_value > 0:
                return connection.prev_node.output_value
            else:
                return 0
        elif self.activation_str == Activation_Functions.SIGMOID:
            return (math.exp(-1 * self.input_value) * connection.prev_node.output_value) / (1 + math.exp(-1 * self.input_value))
        
    def __bias_grad(self):
        if self.activation_str == Activation_Functions.IDENTITY:
            return 1
        elif self.activation_str == Activation_Functions.RELU:
            if self.input_value > 0:
                return 1
            else:
                return 0
        elif self.activation_str == Activation_Functions.SIGMOID:
            return (math.exp(-1 * self.input_value)) / (1 + math.exp(-1 * self.input_value))
    
    def back(self, grad, learning_rate, conn_gradient, bias_gradient):
        #at input values
        if len(self.in_connections) == 0:
            return
        
        #update bias values using bias_grad method to get gradient of node wrt bias
        new_value = grad * self.__bias_grad() * learning_rate
        bias_gradient[self.network_index] += grad * self.__bias_grad() * learning_rate

        for connection in self.in_connections:
            conn_gradient[connection.network_index] += grad * self.__weight_grad(connection) * learning_rate
            connection.prev_node.back(self.__node_grad(connection), learning_rate, conn_gradient, bias_gradient)


class Layer:
    #Special class of nodes, essentially a list of nodes
    #Will need prev layer and next layer
    def __init__(self, node_num, prev_layer, activation_function:Activation_Functions=Activation_Functions.RELU):
        self.prev_layer = prev_layer
        self.nodes = []
        if prev_layer == None:
            for _ in range(node_num):
                self.nodes.append(Node(Activation_Functions.IDENTITY))            
        else:
            for _ in range(node_num):
                self.nodes.append(Node(activation_function))

class Connection:
    #Connection will need weight, bias, prev layer, and next layer
    def __init__(self, prev_node:Node, next_node:Node, network_index:int) -> None:
        self.prev_node = prev_node
        self.next_node = next_node
        self.weight = random.random()
        #self.weight = 2
        self.value = prev_node.output_value
        self.network_index = network_index

class NeuralNetwork:
    def __init__(self, layers:list) -> None:
        self.layers = layers
        self.input_layer = self.layers[0]
        self.output_layer = self.layers[len(self.layers) - 1]

        self.connections = []
        conn_num = 0

        self.nodes = []
        node_num = 0

        for i in range(1,len(self.layers)):
            layer = self.layers[i]
            prev_layer = layer.prev_layer
            for j in layer.nodes:
                j.network_index = node_num
                self.nodes.append(j)
                node_num += 1
                for k in prev_layer.nodes:
                    new_connection = Connection(k, j, conn_num)
                    self.connections.append(new_connection)
                    j.in_connections.append(new_connection)
                    k.out_connections.append(new_connection)
                    conn_num += 1        
    
    def train(self, X:pd.DataFrame, y:pd.DataFrame, epochs, learning_rate):
        for e in range(epochs):
            #print(f'\nEpoch {e}')
            #list of vectors from each training data sample
            out_vectors = []
            out_sums = []
            for i in range(len(X)):
                #evaluate output
                #print('input layer')
                for j in range(len(X.columns)):
                    #print(X.iloc[i][j])
                    self.input_layer.nodes[j].input_value = X.iloc[i][j]
                    self.input_layer.nodes[j].evaluate()
                
                for j in range(1, len(self.layers)):
                    for node in self.layers[j].nodes:
                        node.evaluate()
                        for out_connection in node.out_connections:
                            out_connection.value = node.output_value
                
                #list of output values from a training sample
                out_vector = []
                out_sum = 0
                softmax_sum = 0
                for node in self.output_layer.nodes:
                    out_vector.append(node.output_value)
                    out_sum += node.output_value
                    softmax_sum += math.exp(node.output_value)

                if out_sum > 1:
                    for i in range(len(out_vector)):
                        out_vector[i] = math.exp(out_vector[i]) / softmax_sum

                out_vectors.append(out_vector)
                out_sums.append(out_sum)

            #calculate losses - backpropogation
            loss = 0            
            for i in range(len(out_vectors)):
                R = 0
                for j in range(len(out_vectors[i])):
                    R += out_vectors[i][j] * y.iloc[i][j]
                #print(f'R:{R}')
                loss += math.exp(-R)
            average_loss = loss / len(out_vectors)
            #print(f'loss: {loss}')
            #print(f'average_loss: {average_loss}')

            conn_gradient = np.zeros(len(self.connections))
            bias_gradient = np.zeros(len(self.nodes))
            for i in range(len(out_vectors)):
                if out_sums[i] > 1:
                    for j in range(len(out_vectors[i])):
                        grad = out_vectors[i][j] * -1 * average_loss
                        for k in range(len(self.output_layer.nodes)):
                            if j == k:
                                grad *= out_vectors[i][j] * (1 - out_vectors[i][j])
                            else:
                                grad *= -1 * out_vectors[i][j] * out_vectors[i][k]
                            self.output_layer.nodes[k].back(grad, learning_rate, conn_gradient, bias_gradient)
                else:
                    for j in range(len(out_vectors[i])):
                        grad = out_vectors[i][j] * -1 * average_loss
                        #print(f'gradient: {grad}')
                        self.output_layer.nodes[j].back(grad, learning_rate, conn_gradient, bias_gradient)
            
            for i in range(len(self.connections)):
                self.connections[i].weight -= conn_gradient[i]
            
            for i in range(len(self.nodes)):
                self.nodes[i].bias -= bias_gradient[i]

            # percent = ("{0:." + str(1) + "f}").format(100 * ((e + 1) / float(epochs)))
            # filledLength = int(100 * (e + 1) // epochs)
            # bar = '█' * filledLength + '-' * (100 - filledLength)
            # print(f'\r{e+1}/{epochs} |{bar}| {percent}% complete', end='\r')
            print(f'Epoch {e+1}/{epochs} - Average return: {math.log(average_loss) * -1}')
        #print()
    
    def predict(self, X:pd.DataFrame) -> pd.DataFrame:
        columns = []
        for i in range(len(self.output_layer.nodes)):
            columns.append(f'Column_{i}')
        prediction = pd.DataFrame(columns=columns)
        
        for i in range(len(X)):
            for j in range(len(X.columns)):
                self.input_layer.nodes[j].input_value = X.iloc[i][j]
                self.input_layer.nodes[j].evaluate()
            for j in range(1, len(self.layers)):
                for node in self.layers[j].nodes:
                    node.evaluate()
                    for out_connection in node.out_connections:
                        out_connection.value = node.output_value
            out_vector = []
            out_sum = 0
            softmax_sum = 0
            for node in self.output_layer.nodes:
                out_vector.append(node.output_value)
                out_sum += node.output_value
                softmax_sum += math.exp(node.output_value)
            
            if out_sum > 1:
                    for i in range(len(out_vector)):
                        out_vector[i] = math.exp(out_vector[i]) / softmax_sum
            
            prediction.loc[len(prediction.index)] = out_vector
        
        return prediction



                
# layer_1 = Layer(2, None)
# layer_2 = Layer(16, layer_1)
# layer_3 = Layer(16, layer_2)
# layer_4 = Layer(2, layer_3, Activation_Functions.RELU)

# neural_network = NeuralNetwork([layer_1, layer_2, layer_3, layer_4])

# X_train = pd.DataFrame(data={'col_1':[0.3,0.5,1], 'col_2':[0.1, 0.1, 0]})
# y_train = pd.DataFrame(data={'AAPL':[-0.5,0.1,-0.6], 'MSFT':[0.2, 0.05, -0.1]})
# neural_network.train(X_train, y_train, 3, 0.0001)

# X_test = pd.DataFrame(data={'col_1':[0.2,0.3,0.3], 'col_2':[0.5, -0.1, 0.2]})

# print(neural_network.predict(X_test))

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

input_layer = Layer(len(training_data_df.columns), None)
layer_1 = Layer(32, input_layer)
layer_2 = Layer(32, layer_1)
output_layer = Layer(len(actual_train_df.columns), layer_2)

neural_network = NeuralNetwork(layers=[input_layer, layer_1, layer_2, output_layer])
neural_network.train(training_data_df, actual_train_df, 10, 0.01)

val_data_predictions = neural_network.predict(val_data_df)
val_returns = []
R = 0
for i in range(len(val_data_predictions)):
    r = 0
    for j in range(len(val_data_predictions.columns)):
        r += val_data_predictions.iloc[i][j] * actual_val_df.iloc[i][j]
        R += r * 100
    val_returns.append(R)

plt.plot(val_returns)
plt.show()