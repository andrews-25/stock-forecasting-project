from data_handler import LSTMDataHandler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  #type: ignore




# Config (match your training config where necessary)
ticker = 'AAPL'
config = {
    'seed': 100,
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,  # 70% train, 15% val, 15% test
    'window_size': 30,
    'threshold': None # Not used here, but can be set if needed
}

# Initialize data handler
data_handler = LSTMDataHandler(ticker, config)

# Load raw (unnormalized) test data
(_, X_seq_test, _, X_open_test, _, y_test), _ = data_handler.prepare_data(normalize=False)

# For signal generation
# X_seq_test: for regression input
# X_open_test: today's open price
# y_test: actual next-day close, used later for evaluation

#Predct next close price
model = tf.keras.models.load_model('lstm_close_model.h5', compile=False)
predicted_close = model.predict([X_seq_test, X_open_test])

#Predict next close direction
model = tf.keras.models.load_model('lstm_binary_model.h5', compile=False)
predicted_direction = model.predict([X_seq_test, X_open_test])

# Convert predictions to binary signals
