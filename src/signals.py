from lstm_regression import config
from data_handler import LSTMDataHandler
from tensorflow.keras.models import load_model  # type: ignore
import random
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def regression_signals():
    ticker = input("Enter Ticker: ").upper()

    # Load and prepare data
    data_handler = LSTMDataHandler(ticker, config, target_type='regression')
    (X_seq_test, X_open_test, y_test), scaler = data_handler.prepare_data(training=False)

    model = load_model(f'{ticker}_regression_model.h5', compile=False)
    predicted_close = model.predict([X_seq_test, X_open_test])

    olhcv_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    close_index = olhcv_features.index('Close')
    open_index = olhcv_features.index('Open')

    # Prepare arrays for inverse scaling
    pred_padded = np.zeros((len(predicted_close), len(olhcv_features)))
    true_padded = np.zeros((len(y_test), len(olhcv_features)))
    open_padded = np.zeros((len(X_open_test), len(olhcv_features)))

    pred_padded[:, close_index] = predicted_close.flatten()
    true_padded[:, close_index] = y_test.flatten()
    open_padded[:, open_index] = X_open_test.flatten()

    predicted_next_close = scaler.inverse_transform(pred_padded)[:, close_index]
    next_close = scaler.inverse_transform(true_padded)[:, close_index]
    current_open = scaler.inverse_transform(open_padded)[:, open_index]

    # Generate signals
    signals = np.where(predicted_next_close > current_open, 1, -1)

    # Measure success rate
    successes = np.sum(
        ((signals == 1) & (next_close > current_open)) |
        ((signals == -1) & (next_close < current_open))
    )
    success_rate = successes / len(signals)

    # Calculate actual return from open to close
    returns = (next_close - current_open) / current_open
    strategy_returns = returns * signals

    # Compute metrics
    cumulative_returns = np.cumsum(strategy_returns)
    avg_return = strategy_returns.mean()
    std_return = strategy_returns.std()
    sharpe_ratio = avg_return / std_return if std_return != 0 else 0

    # Print performance summary
    print(f"\nPerformance on {ticker}:")
    print(f"Success Rate: {success_rate * 100:.2f}%")
    print(f"Mean Return per Trade: {avg_return:.4f}")
    print(f"Std of Returns: {std_return:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")




if __name__ == "__main__":
    regression_signals()
