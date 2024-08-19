import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from typeguard import typechecked
from typing import Tuple


# Automatic download of stock price
# own API key for Alpha Vantage: 306SNP4A1M8AB5K4
@typechecked
def get_data(stock: str) -> Tuple[    
    pd.DataFrame,
    pd.DataFrame]:
    """
    args:
        stock: abbreviation of the stock for downloading the data from Alpha Vantage
    
    returns:
        full_data: full pandas data frame including the complete csv file downloaded from Alpha Vantage for the specified stock
        closing_prices: pandas data frame including the closing prices for the specified stock
    """  
    # url="https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=" + stock + "&outputsize=full&apikey=306SNP4A1M8AB5K4&datatype=csv"
    # data=pd.read_csv(url)
    
    full_data = pd.read_csv("./data/daily_IBM.csv")
    # Reverse the data to get top -> down (old -> new) ordering
    full_data = full_data.iloc[::-1].reset_index(drop=True)
    
    # Take only the closing prices
    data = full_data[['close']].astype('float32')



    return full_data, data


@typechecked
def train_test_split(data: pd.DataFrame) -> Tuple[
    pd.DataFrame,
    pd.DataFrame]:
    """
    args: 
        data: pandas data frame including the closing prices

    returns:
        train_set: 80% of the complete data set for training purpose
        test_set: 20% of the complete data set for test purpose
    """
    no_datapoints = data.shape[0]
    
    # Plit into 80% train set, 20% test set
    train_set_end_index = math.ceil(no_datapoints*0.8)
    
    train_set = data.iloc[:train_set_end_index, :]
    test_set = data.iloc[train_set_end_index:, :]

    return train_set, test_set


@typechecked
def get_scaled_closing_prices(data: pd.DataFrame) -> Tuple[
    pd.DataFrame,
    float]:
    """
    args: 
        data: pandas data frame including the complete csv file downloaded from Alpha Vantage 
        scaler: MinMaxScaler
    returns:
        scaled_close: closing prices scaled 0-1 range 
        maximum_value: highest closing price by which we divide all the other closing prices to scale down to 0 - 1
    """
    maximum_value = data['close'].max()
    scaled_closing_prices = data / maximum_value
    return scaled_closing_prices, maximum_value


# Prepare the input and output sequences as np.arrays based on the data from the csv file
# The input sequence contains all elements in a window before the element of interest with window size sequence_length
@typechecked
def create_sequences(data: pd.DataFrame, window_size: int) -> Tuple[
    np.ndarray,
    np.ndarray]:
    """
    args: 
        data: pandas data frame including the closing prices
        window_size: window size of input sequence which is of relevance for the prediction of the next output
    returns:
        X: input sequence, i.e. the last sequence_length data points which are relevant for the prediction of the next one
        y: output 
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        # Extract the input sequence and ensure it is a NumPy array
        X.append(data.iloc[i:(i+window_size)].to_numpy())
        # Extract the output value and ensure it is a scalar value, not a Series
        y.append(data.iloc[i+window_size].to_numpy())
    
    # Convert lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y

@typechecked
def show_chart(full_data: pd.DataFrame, testPredictPlot: np.ndarray, stock: str):
    """
    args:
        data: full pandas data frame including the complete csv file downloaded from Alpha Vantage 
    
    returns:
        -
    """
    no_datapoints = full_data.shape[0]
    plt.plot(full_data['timestamp'], full_data['close'], linestyle = 'solid')
    plt.plot(testPredictPlot)
    plt.title("Closing Prices for the last " + str(no_datapoints) + " days for " + stock)
    plt.xticks([1, math.ceil(no_datapoints/4), math.ceil(no_datapoints/2), math.ceil(3*no_datapoints/4), no_datapoints-1])
    plt.show()
    
# Plot the predictions on the training and test data set
@typechecked
def plot_test_predictions(full_data: pd.DataFrame, data: pd.DataFrame, testPredictions: np.ndarray, stock: str, window_size: int):
    """
    args:
        data: full data set of closing prices
        testPredictions: results from the prediction on the test data
        len_trainPredict: number of days used for training
    returns:
        -
    """  
    len_trainPredict = data.shape[0] - len(testPredictions)
    
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(data)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len_trainPredict:len_trainPredict+len(testPredictions) + (window_size*2), :] = testPredictions
    
    # plot baseline and predictions
    show_chart(full_data, testPredictPlot, stock)
