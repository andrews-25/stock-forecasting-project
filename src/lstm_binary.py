import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import tensorflow as tf
from tensorflow.keras.models import Model            # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate # type: ignore
from tensorflow.keras.optimizers import Adam                # type: ignore
from tensorflow.keras.losses import binary_crossentropy                     # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from data_handler import LSTMDataHandler
import matplotlib.pyplot as plt

config = {
    'seed': 100,
    'train_split': 0.8,            # 80% train, 20% test
    'window_size': 20,             # sequence length for LSTM input
    'lstm_units_1': 128,
    'lstm_units_2': 64,
    'dropout_rate_1': 0.1,
    'dropout_rate_2': 0.1,
    'dense_units_open': 32,
    'merge_dense_units': 16,
    'merge_dropout_rate': 0.1,
    'learning_rate': 0.001,
    'batch_size': 8,
    'epochs': 1000,
    'early_stopping_patience': 20,
    'early_stopping_min_delta': 0.00005,
    'reduce_lr_factor': 0.5,
    'reduce_lr_patience': 10,
    'reduce_lr_min_lr': 1e-6,
}

#Set seed and device check
random.seed(config['seed']  )
np.random.seed(config['seed'])
tf.random.set_seed(config['seed'])
os.environ['TF_DETERMINISTIC_OPS'] = '1'
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("Using device:", "GPU" if gpus else "CPU")

# User input ticker
ticker = input("Enter Ticker: ").upper()


data_handler = LSTMDataHandler(ticker, config, target_type='regression')
(X_seq_train, X_seq_test, X_open_train, X_open_test, y_train, y_test), scaler = data_handler.prepare_data()
features = data_handler.features
window = config['window_size']

#### BUILDING THE MODEL ####

#lstm layers for yesterdays OLHCV
lstm_input = Input(shape=(window, len(features)), name='lstm_input')
lstm_layer1 = LSTM(config['lstm_units_1'], return_sequences=True, name='lstm_layer1')(lstm_input)
lstm_layer1_dropout = Dropout(config['dropout_rate_1'], name='dropout1')(lstm_layer1)
lstm_out_2 = LSTM(config['lstm_units_2'], return_sequences=False, name='lstm_layer12')(lstm_layer1_dropout)
lstm_layer2_dropout = Dropout(config['dropout_rate_2'], name='dropout2')(lstm_out_2)

#Dense layer for today's open price
open_input = Input(shape=(1,), name='open_input')
open_dense = Dense(config['dense_units_open'], activation='relu', name='dense_open')(open_input)

#merge the outputs
concat = Concatenate(name='concat')([lstm_layer2_dropout, open_dense])
merge = Dense(config['merge_dense_units'], activation='relu', name='dense_concat')(concat)
merge_dropout = Dropout(config['merge_dropout_rate'], name='dropout_merge')(merge)

#output layer
output = Dense(1, activation = 'sigmoid', name='output')(merge_dropout)

model = Model(inputs=[lstm_input, open_input], outputs=output)

model.compile(
    optimizer=Adam(learning_rate=config['learning_rate']),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)
model.summary()