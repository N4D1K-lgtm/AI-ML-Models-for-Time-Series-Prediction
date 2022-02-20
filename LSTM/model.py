import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import tensorflow as tf
import IPython
import IPython.display
from lstmlib import WindowGenerator, compile_and_fit, FeedBack

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Masking
from sklearn.preprocessing import MinMaxScaler

dataset = pd.DataFrame()
train_df = pd.DataFrame()
val_df = pd.DataFrame()
test_df = pd.DataFrame()

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

def main():
    global dataset 
    global train_df
    global val_df
    global test_df

    CONV_WIDTH = 3
    
    path = "/home/k1/freqtrade/user_data/data/binanceus/"
    files = os.listdir(path)

    for file in files:
        
        name = str(file)
        name = name.replace(".json", "")
        df = pd.read_json(path + str(file))
        df = df.rename(columns = {0 : 'Time', 1 : 'Open', 2 : 'High', 3 : 'Low', 4 : 'Close', 5 : 'Volume'})
        df.index.name = 'index'
        df['OCAvg'] = df[['Open', 'Close']].mean(axis=1)
        series = pd.Series(df['OCAvg'])
        dataset[name] = series

    dataset = dataset.dropna(axis=0)

    # Feature Scaling
    scaler = MinMaxScaler(feature_range = (0,1))
    scaled = scaler.fit_transform(dataset.loc[:])
    scaled_dataset = pd.DataFrame(scaled, columns=dataset.columns, index=dataset.index)
    
    
    # Split into Train, Validation and Test Sets.
    column_indices = {name: i for i, name in enumerate(scaled_dataset.columns)}
    plot_cols = scaled_dataset.columns[:]
    plot_features = scaled_dataset[plot_cols]

    n = len(scaled_dataset)
    train_df = scaled_dataset[0:int(n*0.7)]
    val_df = scaled_dataset[int(n*0.7):int(n*0.9)]
    test_df = scaled_dataset[int(n*0.9):]

    num_features = scaled_dataset.shape[1]

    wide_window = WindowGenerator(32, 1, 1, train_df, val_df, test_df)

    feedback_model = FeedBack(units=32, out_steps=1, num_features=num_features)
    
    print('Output shape (batch, time, features): ', feedback_model(wide_window.example[0]).shape)

    prediction, state = feedback_model.warmup(wide_window.example[0])
    
    feedback_model.summary()

    history = compile_and_fit(feedback_model, wide_window, 20)
    IPython.display.clear_output()

    performance = {}
    val_performance = {}

    val_performance['AR LSTM'] = feedback_model.evaluate(wide_window.val)
    performance['AR LSTM'] = feedback_model.evaluate(wide_window.test, verbose=0)
    wide_window.plot(feedback_model, plot_col=train_df.columns)
    
    

if __name__ == "__main__":
    
    main()
