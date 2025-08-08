import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import tensorflow as tf
from tensorflow.keras.models import Model   # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.losses import Huber  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from data_handler import DataHandler
from tensorflow.keras.regularizers import l2  # type: ignore
import sys

config = {
    'seed': 100,
    'train_split': 0.7,
    'val_split': .15,
    'test_split': .15,
    'window_size': 20,
    'lstm_units_1': 128,
    'lstm_units_2': 64,
    'lstm_units_3': 32,
    'dense_units_open': 32, 
    'dense_units_concat': 16,
    'dropout_rate_1': 0.25,
    'dropout_rate_2': 0.25,
    'dropout_rate_3': 0.25,
    'dropout_rate_final': 0.25,
    'learning_rate': 0.008,
    'batch_size': 8,
    'epochs': 1000,
    'regularization_strength': 5e-4,
    'early_stopping_patience': 30,
    'early_stopping_min_delta': 0.00002,
    'reduce_lr_factor': 0.5,
    'reduce_lr_patience': 10,
    'reduce_lr_min_lr': 1e-6,
}

def build_model(window, features, config):
    lstm_input = Input(shape=(window, len(features)), name='lstm_input')
    x = LSTM(config['lstm_units_1'], return_sequences=True, kernel_regularizer=l2(config['regularization_strength']))(lstm_input)
    x = Dropout(config['dropout_rate_1'])(x)
    x = LSTM(config['lstm_units_2'], return_sequences=True, kernel_regularizer=l2(config['regularization_strength']))(x)
    x = Dropout(config['dropout_rate_2'])(x)
    x = LSTM(config['lstm_units_3'], return_sequences=False, kernel_regularizer=l2(config['regularization_strength']))(x)
    x = Dropout(config['dropout_rate_3'])(x)

    current_open_input = Input(shape=(1,), name='model_input_open')
    dense_open = Dense(config['dense_units_open'], activation='relu')(current_open_input)
    concat = Concatenate()([x, dense_open])

    dense = Dense(config['dense_units_concat'], activation='relu')(concat)
    dense = Dropout(config['dropout_rate_final'])(dense)
    output = Dense(1)(dense)

    model = Model(inputs=[lstm_input, current_open_input], outputs=output)

    opt = Adam(learning_rate=config['learning_rate'])
    model.compile(opt, loss=Huber(delta=1.0), metrics=['mae'])
    return model

def train_and_evaluate(ticker, config):
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    tf.random.set_seed(config['seed'])
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    data_handler = DataHandler(ticker, config)

    X_seq_train, X_open_train, y_train, X_seq_val, X_open_val, y_val, val_dates = data_handler.prepare_data()
    features = data_handler.featurelist
    window = config['window_size']

    model = build_model(window, features, config)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config['early_stopping_patience'],
            restore_best_weights=True,
            min_delta=config['early_stopping_min_delta'],
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=config['reduce_lr_factor'],
            patience=config['reduce_lr_patience'],
            verbose=1,
            min_lr=config['reduce_lr_min_lr']
        )
    ]

    history = model.fit(
        [X_seq_train, X_open_train],
        y_train,
        validation_data=([X_seq_val, X_open_val], y_val),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        verbose=1,
        callbacks=callbacks
    )

    model.save(f'{ticker}_regression_model.h5')

    predicted_close = model.predict([X_seq_val, X_open_val])

    predicted_close = predicted_close.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)

    feature_scaler, open_scaler, close_scaler = data_handler.get_scalers()

    predicted_close_real = close_scaler.inverse_transform(predicted_close).flatten()
    y_val_real = close_scaler.inverse_transform(y_val).flatten()

    mse = mean_squared_error(y_val_real, predicted_close_real)
    mae = mean_absolute_error(y_val_real, predicted_close_real)
    r2 = r2_score(y_val_real, predicted_close_real)
    avg_daily_change = np.mean(np.abs(np.diff(y_val_real)))
    std_change = np.std(np.diff(y_val_real))

    print("Sample predictions vs actual (denormalized):")
    for i in range(min(5, len(predicted_close_real))):
        print(f"Predicted: ${predicted_close_real[i]:.2f} | Actual: ${y_val_real[i]:.2f}")

    print(f"\nModel Performance:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R Squared Score: {r2:.4f}")
    print(f"Average Daily Change: {avg_daily_change:.4f}")
    print(f"Standard Deviation of Daily Change: {std_change:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Huber)')
    plt.legend()
    plt.grid(True)
    plt.show()

    val_dates = val_dates[:len(y_val_real)]

    plt.figure(figsize=(10, 6))
    plt.plot(val_dates, y_val_real, label='Actual Close Price', color='blue')
    plt.plot(val_dates, predicted_close_real, label='Predicted Close Price', color='orange')
    plt.title('Actual vs Predicted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ticker = input("Enter Ticker: ").upper()
    train_and_evaluate(ticker, config)
