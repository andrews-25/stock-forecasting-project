# === Imports ===
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model            # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate # type: ignore
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
target_close = [] #Todays Close (Target)

#Fill input and target sequences to train on
for i in range (window, len(df)):
    sequence = df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[i-window:i].values
    current_open = df.iloc[i]['Open']
    current_close = df.iloc[i]['Open']
    
    input_seq.append(sequence)
    input_open.append([current_open])
    target_close.append(current_close)


#Data Prep
X_seq = np.array(input_seq)
X_open = np.array(input_open)
y = np.array(target_close)
X_seq_train, X_seq_test, X_open_train, X_open_test, y_train, y_test = train_test_split(X_seq, X_open, y, test_size = 0.2, shuffle = False)

lstm_input = Input(shape=(window, 5), name = 'lstm_input')
lstm_output = LSTM(64, return_sequences = False, name = 'lstm_layer')(lstm_input)
lstm_output_dropout = Dropout(0.2, name = 'lstm_dropout')(lstm_output)

model_input_open = Input(shape=(1,), name = 'model_input_open')
dense_open = Dense(16, activation = 'relu', name='open_dense')(model_input_open)
concat = Concatenate(name = 'concat')([lstm_output_dropout, dense_open])

model_dense_1 = Dense(64, activation='relu', name='dense_1')(concat)
model_dropout_2 = Dropout(0.2, name = 'model_dropout')(model_dense_1)
model_output = Dense(1, name = 'model_output')(model_dropout_2)

model = Model(inputs = [lstm_input, model_input_open], outputs = model_output, name = 'ClosePredictorLSTM')
model.summary()

