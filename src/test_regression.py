from lstm_regression import config
from data_handler import LSTMDataHandler
from tensorflow.keras.models import load_model #type: ignore
import random
import os
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def regression_predictions():
    # User input ticker
    ticker = input("Enter Ticker: ").upper()

    # Load and prepare data
    data_handler = LSTMDataHandler(ticker, config, target_type='regression')
    (X_seq_test, X_open_test, y_test), scaler = data_handler.prepare_data(training=False)
    (X_seq_train, X_seq_val, X_open_train, X_open_val, y_train, y_val), scaler = data_handler.prepare_data()
    features = data_handler.features
    window = config['window_size']


    model = load_model(f'{ticker}_regression_model.h5', compile=False)
    predicted_close = model.predict([X_seq_test, X_open_test])
    
    olhcv_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    close_index = olhcv_features.index('Close')

    pred_padded = np.zeros((len(predicted_close), len(olhcv_features)))
    true_padded = np.zeros((len(y_test), len(olhcv_features)))

    pred_padded[:, close_index] = predicted_close.flatten()
    true_padded[:, close_index] = y_test.flatten()

    predicted_close_real = scaler.inverse_transform(pred_padded)[:, close_index]
    y_test_real = scaler.inverse_transform(true_padded)[:, close_index]

    mse = mean_squared_error(y_test_real, predicted_close_real)
    mae = mean_absolute_error(y_test_real, predicted_close_real)
    r2 = r2_score(y_test_real, predicted_close_real)
    avg_daily_change = np.mean(np.abs(np.diff(y_test_real)))
    std_change = np.std(np.diff(y_test_real))

    
    print(f"\nModel Performance:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R Squared Score: {r2:.4f}")
    print(f"Average Daily Change: {avg_daily_change:.4f}")
    print(f"Standard Deviation of Daily Change: {std_change:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(y_test_real, label='Actual Close Prices', color='blue')
    plt.plot(predicted_close_real, label='Predicted Close Prices', color='orange')
    plt.title(f'Actual vs Predicted Close Prices for {ticker}')
    plt.xlabel('Time Steps')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    regression_predictions()