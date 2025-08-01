# === Imports ===
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import tensorflow as tf
from tensorflow.keras.models import Model            # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate # type: ignore
from tensorflow.keras.optimizers import Adam                # type: ignore
from tensorflow.keras.losses import Huber                     # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from data_handler import LSTMDataHandler
import matplotlib.pyplot as plt
#Hyperparameters
config = {
    'seed': 100,
    'train_split': 0.8,            # 80% train, 20% test
    'window_size': 20,             # sequence length for LSTM input
    'lstm_units_1': 128,
    'lstm_units_2': 64,
    'dropout_rate_1': 0.1,
    'dropout_rate_2': 0.1,
    'dense_units_open': 32,
    'dense_units_concat': 16,
    'dropout_rate_final': 0.1,
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

# Load and prepare data
data_handler = LSTMDataHandler(ticker, config, target_type='regression')
(X_seq_train, X_seq_test, X_open_train, X_open_test, y_train, y_test), scaler = data_handler.prepare_data()
features = data_handler.features
window = config['window_size']





#### BUILDING THE MODEL ####
lstm_input = Input(shape=(window, len(features)), name='lstm_input')
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


#### TRAINING THE MODEL ####
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



#### EVALUATING THE MODEL####
predicted_close = model.predict([X_seq_test, X_open_test])
close_index = features.index('Close')

pred_padded = np.zeros((len(predicted_close), len(features)))
true_padded = np.zeros((len(y_test), len(features)))

pred_padded[:, close_index] = predicted_close.flatten()
true_padded[:, close_index] = y_test.flatten()

predicted_close_real = scaler.inverse_transform(pred_padded)[:, close_index]
y_test_real = scaler.inverse_transform(true_padded)[:, close_index]

#Calculate metrics
mse = mean_squared_error(y_test_real, predicted_close_real)
mae = mean_absolute_error(y_test_real, predicted_close_real)
r2 = r2_score(y_test_real, predicted_close_real)
avg_daily_change = np.mean(np.abs(np.diff(y_test_real)))
std_change = np.std(np.diff(y_test_real))


#### PRINT AND PLOT RESULTS ####
print("Sample predictions vs actual (denormalized):")
for i in range(5):
    print(f"Predicted: ${predicted_close_real[i]:.2f} | Actual: ${y_test_real[i]:.2f}")
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
plt.plot(y_test_real, label='Actual Close Price', color='blue')
plt.plot(predicted_close_real, label='Predicted Close Price', color='orange')
plt.title('Actual vs Predicted Close Price')
plt.xlabel('Time Steps')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()
