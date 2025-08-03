from data_handler import LSTMDataHandler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  #type: ignore




# Config (match your training config where necessary)
ticker = 'AAPL'
config = {
    'seed': 100,
    'train_split': 0.8,
    'window_size': 30
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
model = tf.keras.models.load_model('src/lstm_close_model.h5', compile=False)
predicted_close = model.predict([X_seq_test, X_open_test])

#Predict next close direction
model = tf.keras.models.load_model('lstm_binary_model.h5', compile=False)
predicted_direction = model.predict([X_seq_test, X_open_test])

print("Predicted close shape:", predicted_close.shape)
print("Predicted direction shape:", predicted_direction.shape)
print("First 5 predicted directions (probabilities):", predicted_direction[:5].flatten())