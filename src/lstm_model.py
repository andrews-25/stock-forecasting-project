# === Imports ===
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential             # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Concatenate# type: ignore
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from data_handler import get_data



#Device Check
print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("Using device: GPU")
else:
    print("Using device: CPU")
    
### Get Data and Normalize ###
def normalize(df, features):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    scaled_df = pd.DataFrame(scaled, columns = features, index = df.index)
    return  scaled_df, scaler 
ticker = input("Enter Ticker: ").upper()
df = get_data(ticker)
features = ['Open', 'High', 'Low', 'Close', 'Volume']
normalized_df, scaler = normalize(df, features)

window = 30
input_seq = [] #LHCV inputs
input_open = [] #current open
output_close = [] #Todays Close

for i in range (window, len(df)):
    sequence = df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[i-window:i].values
    current_open = df.iloc[i]['Open']
    current_close = df.iloc[i]['Open']
    
    input_seq.append(sequence)
    input_open.append([current_open])
    output_close.append(current_close)

