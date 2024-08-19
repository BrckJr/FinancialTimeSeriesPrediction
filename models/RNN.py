import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error

class RNNModel:
    def __init__(self, window_size, rnn_units, loss='mean_squared_error', optimizer='adam'):
        self.window_size = window_size
        self.rnn_units = rnn_units
        
        # Create the model
        self.model = Sequential()
        self.model.add(tf.keras.Input(shape=(1, window_size)))
        self.model.add(SimpleRNN(rnn_units))
        self.model.add(Dense(1))
        
        # Compile the model
        self.model.compile(loss=loss, optimizer=optimizer)
    
    def train(self, train_X, train_Y, epochs, batch_size, verbose=0):
        self.model.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    
    def predict(self, X):
        return self.model.predict(X, verbose=0)
    
    def summary(self):
        return self.model.summary()

    def save(self, link_to_folder):
        return self.model.save(link_to_folder + '/rnn_model.keras')

    # Calculate the root mean square error between the predicted and underlying value
    def calculateRMS(self, Y, predict):
        score = np.sqrt(mean_squared_error(Y, predict))
        # print('Score: %.2f RMSE' % (score))
        return score

    # Calculate the mean absolute error between the predicted and underlying value
    def calculateMAE(self, Y, predict):
        score = mean_absolute_error(Y, predict)
        # print('Score: %.2f MAE' % (score))
        return score
