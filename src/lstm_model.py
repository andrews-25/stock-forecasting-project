# === Imports ===
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential             # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
import matplotlib.pyplot as plt
from data_handler import get_data



#Device Check
print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("Using device: GPU")
else:
    print("Using device: CPU")
def normalize(df, features):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    scaled_df = pd.DataFrame(scaled, columns = features, index = df.index)
    return  scaled_df, scaler 
ticker = input("Enter Ticker: ")
df = get_data(ticker)
features = ['Open', 'High', 'Low', 'Close', 'Volume']
normalized_df, scaler = normalize(df, features)
print(normalized_df.describe())
