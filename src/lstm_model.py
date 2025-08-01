# === Imports ===
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model            # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate # type: ignore
from tensorflow.keras.optimizers import Adam                #type: ignore
from tensorflow.keras.losses import Huber                     # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from data_handler import get_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random

# === Hyperparameters ===
config = {
    'seed': 100,
    'train_split': 0.8,            # 80% train, 20% test
    'window_size': 30,             # sequence length for LSTM input
    'lstm_units_1': 128,
    'lstm_units_2': 64,
    'dropout_rate_1': 0.2,
    'dropout_rate_2': 0.2,
    'dense_units_open': 32,
    'dense_units_concat': 16,
    'dropout_rate_final': 0.2,
    'learning_rate': 0.001,
    'batch_size': 8,
    'epochs': 1000,
    'early_stopping_patience': 20,
    'early_stopping_min_delta': 0.00005,
    'reduce_lr_factor': 0.5,
    'reduce_lr_patience': 10,
    'reduce_lr_min_lr': 1e-6,
}

# Set seeds for reproducibility
random.seed(config['seed'])
np.random.seed(config['seed'])
tf.random.set_seed(config['seed'])
os.environ['TF_DETERMINISTIC_OPS'] = '1'  # For more determinism on GPU

# Device Check
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("Using device: GPU")
else:
    print("Using device: CPU")

### Get Data and Normalize ###
def normalize(df, features):
    split_index = int(len(df) * config['train_split'])
    train_df = df.iloc[:split_index]

    scaler = MinMaxScaler()
    scaler.fit(train_df[features])

    scaled = scaler.transform(df[features])
    scaled_df = pd.DataFrame(scaled, columns=features, index=df.index)
    return scaled_df, scaler

ticker = input("Enter Ticker: ").upper()
df = get_data(ticker)
features = ['Open', 'High', 'Low', 'Close', 'Volume']
normalized_df, scaler = normalize(df, features)

window = config['window_size']
input_seq = []    # LHCV inputs
input_open = []   # current open
target_close = [] # today's close (target)

# Fill input and target sequences
for i in range(window + 1, len(df)):
    sequence = normalized_df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[(i - 1) - window:(i - 1)].values
    current_open = normalized_df.iloc[i]['Open']
    current_close = normalized_df.iloc[i]['Close']

    input_seq.append(sequence)
    input_open.append([current_open])
    target_close.append(current_close)

# Data Prep
X_seq = np.array(input_seq)
X_open = np.array(input_open)
y = np.array(target_close)
test_size = 1 - config['train_split']
X_seq_train, X_seq_test, X_open_train, X_open_test, y_train, y_test = train_test_split(
    X_seq, X_open, y, test_size=test_size, shuffle=False
)

# Build model
lstm_input = Input(shape=(window, 5), name='lstm_input')
lstm_layer1 = LSTM(config['lstm_units_1'], return_sequences=True, name='lstm_layer_1')(lstm_input)
lstm_layer1_dropout = Dropout(config['dropout_rate_1'], name='lstm_dropout')(lstm_layer1)
lstm_layer2 = LSTM(config['lstm_units_2'], return_sequences=False, name='lstm_layer_2')(lstm_layer1_dropout)
lstm_layer2_dropout = Dropout(config['dropout_rate_2'], name='lstm_layer2_dropout')(lstm_layer2)

current_open_input = Input(shape=(1,), name='model_input_open')
dense_open = Dense(config['dense_units_open'], activation='relu', name='open_dense')(current_open_input)
concat = Concatenate(name='concat')([lstm_layer2_dropout, dense_open])

model_dense_1 = Dense(config['dense_units_concat'], activation='relu', name='dense_1')(concat)
model_dropout_2 = Dropout(config['dropout_rate_final'], name='model_dropout')(model_dense_1)
model_output = Dense(1, name='model_output')(model_dropout_2)

model = Model(inputs=[lstm_input, current_open_input], outputs=model_output, name='ClosePredictorLSTM')

opt = Adam(learning_rate=config['learning_rate'])
model.compile(opt, loss=Huber(delta=1.0), metrics=['mae'])

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=config['early_stopping_patience'],
    restore_best_weights=True,
    min_delta=config['early_stopping_min_delta'],
    verbose=1
)

learning_rate = ReduceLROnPlateau(
    monitor='val_loss',
    factor=config['reduce_lr_factor'],
    patience=config['reduce_lr_patience'],
    verbose=1,
    min_lr=config['reduce_lr_min_lr']
)

history = model.fit(
    [X_seq_train, X_open_train],
    y_train,
    validation_data=([X_seq_test, X_open_test], y_test),
    epochs=config['epochs'],
    batch_size=config['batch_size'],
    verbose=1,
    callbacks=[early_stopping, learning_rate]
)

### Running the Model ###
predicted_close = model.predict([X_seq_test, X_open_test])
close_index = features.index('Close')

pred_padded = np.zeros((len(predicted_close), len(features)))
true_padded = np.zeros((len(y_test), len(features)))

pred_padded[:, close_index] = predicted_close.flatten()
true_padded[:, close_index] = y_test.flatten()

# Inverse transform both
predicted_close_real = scaler.inverse_transform(pred_padded)[:, close_index]
y_test_real = scaler.inverse_transform(true_padded)[:, close_index]

# Print a few comparisons
print("Sample predictions vs actual (denormalized):")
for i in range(5):
    print(f"Predicted: ${predicted_close_real[i]:.2f} | Actual: ${y_test_real[i]:.2f}")

# Run stats
mse = mean_squared_error(y_test_real, predicted_close_real)
mae = mean_absolute_error(y_test_real, predicted_close_real)
r2 = r2_score(y_test_real, predicted_close_real)

print(f"\n Model Performance:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R Squared Score: {r2:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(y_test_real, label='Actual Close Price', color='blue')
plt.plot(predicted_close_real, label='Predicted Close Price', color='orange')
plt.title('Actual vs Predicted Close Price')
plt.xlabel('Time Steps')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()
