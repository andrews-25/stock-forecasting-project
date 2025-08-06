import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import tensorflow as tf
from tensorflow.keras.models import Model   #type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate    #type: ignore
from tensorflow.keras.optimizers import Adam    #type: ignore
from tensorflow.keras.losses import Huber   #type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau #type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from data_handler import LSTMDataHandler
from tensorflow.keras.regularizers import l2    #type: ignore
import sys
config = {
    'seed': 100,
    'train_split': 0.7,
    'val_split': .15,
    'test_split': .15,
    'window_size': 30,
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
    'regularization_strength': 0.0005,
    'early_stopping_patience': 30,
    'early_stopping_min_delta': 0.00002,
    'reduce_lr_factor': 0.5,
    'reduce_lr_patience': 10,
    'reduce_lr_min_lr': 1e-6,
}

def build_model(window, features, config):
    lstm_input = Input(shape=(window, len(features)), name='lstm_input')
    lstm_layer1 = LSTM(config['lstm_units_1'], return_sequences=True, kernel_regularizer=l2(config['regularization_strength']), name='lstm_layer_1')(lstm_input)
    lstm_layer1_dropout = Dropout(config['dropout_rate_1'], name='lstm_dropout')(lstm_layer1)
    lstm_layer2 = LSTM(config['lstm_units_2'], return_sequences=True, kernel_regularizer=l2(config['regularization_strength']), name='lstm_layer_2')(lstm_layer1_dropout)
    lstm_layer2_dropout = Dropout(config['dropout_rate_2'], name='lstm_layer2_dropout')(lstm_layer2)
    lstm_layer3 = LSTM(config['lstm_units_3'], return_sequences=False, kernel_regularizer=l2(config['regularization_strength']), name='lstm_layer_3')(lstm_layer2_dropout)
    lstm_layer3_dropout = Dropout(config['dropout_rate_3'], name='lstm_layer3_dropout')(lstm_layer3)

    current_open_input = Input(shape=(1,), name='model_input_open')
    dense_open = Dense(config['dense_units_open'], activation='relu', name='open_dense')(current_open_input)
    concat = Concatenate(name='concat')([lstm_layer3_dropout, dense_open])

    model_dense_1 = Dense(config['dense_units_concat'], activation='relu', name='dense_1')(concat)
    model_dropout_2 = Dropout(config['dropout_rate_final'], name='model_dropout')(model_dense_1)
    model_output = Dense(1, name='model_output')(model_dropout_2)

    model = Model(inputs=[lstm_input, current_open_input], outputs=model_output, name='ClosePredictorLSTM')

    opt = Adam(learning_rate=config['learning_rate'])
    model.compile(opt, loss=Huber(delta=1.0), metrics=['mae'])

    return model

def train_and_evaluate(ticker, config):
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    tf.random.set_seed(config['seed'])
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    print("TensorFlow version:", tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    print("Using device:", "GPU" if gpus else "CPU")

    data_handler = LSTMDataHandler(ticker, config, target_type='regression')
    (X_seq_train, X_seq_val, X_open_train, X_open_val, y_train, y_val), scaler = data_handler.prepare_data()
    features = data_handler.features
    window = config['window_size']

    model = build_model(window, features, config)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config['early_stopping_patience'],
        restore_best_weights=True,
        min_delta=config['early_stopping_min_delta'],
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=config['reduce_lr_factor'],
        patience=config['reduce_lr_patience'],
        verbose=1,
        min_lr=config['reduce_lr_min_lr']
    )

    history = model.fit(
        [X_seq_train, X_open_train],
        y_train,
        validation_data=([X_seq_val, X_open_val], y_val),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        verbose=1,
        callbacks=[early_stopping, reduce_lr]
    )

    model.save(f'{ticker}_regression_model.h5')

    predicted_close = model.predict([X_seq_val, X_open_val])
    olhcv_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_40']
    close_index = olhcv_features.index('Close')

    pred_padded = np.zeros((len(predicted_close), len(olhcv_features)))
    true_padded = np.zeros((len(y_val), len(olhcv_features)))

    pred_padded[:, close_index] = predicted_close.flatten()
    true_padded[:, close_index] = y_val.flatten()

    predicted_close_real = scaler.inverse_transform(pred_padded)[:, close_index]
    y_val_real = scaler.inverse_transform(true_padded)[:, close_index]

    mse = mean_squared_error(y_val_real, predicted_close_real)
    mae = mean_absolute_error(y_val_real, predicted_close_real)
    r2 = r2_score(y_val_real, predicted_close_real)
    avg_daily_change = np.mean(np.abs(np.diff(y_val_real)))
    std_change = np.std(np.diff(y_val_real))

    print("Sample predictions vs actual (denormalized):")
    for i in range(5):
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

    plt.figure(figsize=(10, 6))
    plt.plot(y_val_real, label='Actual Close Price', color='blue')
    plt.plot(predicted_close_real, label='Predicted Close Price', color='orange')
    plt.title('Actual vs Predicted Close Price')
    sys.exit()
    plt.xlabel('Time Steps')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    ticker = input("Enter Ticker: ").upper()
    #build_model()
    train_and_evaluate(ticker, config)

