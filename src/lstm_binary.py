import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import tensorflow as tf
from tensorflow.keras.models import Model            # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate # type: ignore
from tensorflow.keras.optimizers import Adam                # type: ignore
from tensorflow.keras.losses import binary_crossentropy                     # type: ignore
from tensorflow.keras.regularizers import l2  #type: ignore

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
from data_handler import LSTMDataHandler
import matplotlib.pyplot as plt

config = {
    'seed': 100,
    'train_split': 0.8,            # 80% train, 20% test
    'window_size': 30,             # sequence length for LSTM input
    'lstm_units_1': 64,
    'lstm_units_2': 32,
    'dropout_rate_1': 0.2,
    'dropout_rate_2': 0.2,
    'dense_units_open': 16,
    'merge_dense_units': 16,
    'merge_dropout_rate': 0.2,
    'learning_rate': 0.0005,
    'batch_size': 12,
    'epochs': 1000,
    'early_stopping_patience': 20,
    'early_stopping_min_delta': 0.00001,
    'reduce_lr_factor': 0.5,
    'reduce_lr_patience': 10,
    'reduce_lr_min_lr': 1e-6,
    'regularization_strength': 0.001,

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


data_handler = LSTMDataHandler(ticker, config, target_type='classification')
(X_seq_train, X_seq_test, X_open_train, X_open_test, y_train, y_test), scaler = data_handler.prepare_data()
features = data_handler.features
window = config['window_size']

#### BUILDING THE MODEL ####

#lstm layers for yesterdays OLHCV
lstm_input = Input(shape=(window, len(features)),name='lstm_input')
lstm_layer1 = LSTM(config['lstm_units_1'], return_sequences=True, kernel_regularizer = l2(config['regularization_strength']), name='lstm_layer1')(lstm_input)
lstm_layer1_dropout = Dropout(config['dropout_rate_1'], name='dropout1')(lstm_layer1)
lstm_out_2 = LSTM(config['lstm_units_2'], return_sequences=False, kernel_regularizer = l2(config['regularization_strength']), name='lstm_layer12')(lstm_layer1_dropout)
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
    loss=binary_crossentropy,
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)
#### TRAINING THE MODEL ####
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=config['early_stopping_patience'],
    min_delta=config['early_stopping_min_delta'],
    restore_best_weights=True
)
learning_rate = ReduceLROnPlateau(
    monitor='val_loss',
    factor=config['reduce_lr_factor'],
    patience=config['reduce_lr_patience'],
    min_lr=config['reduce_lr_min_lr'],
    verbose=1
)
history = model.fit(
    [X_seq_train, X_open_train],
    y_train,
    validation_data=([X_seq_test, X_open_test], y_test),
    epochs=config['epochs'],
    batch_size=config['batch_size'],
    callbacks=[early_stopping, learning_rate],
    verbose=1
)


### MODEL EVALUATION ###

# Predict probabilities (assuming model outputs shape (n_samples, 1))
y_pred = model.predict([X_seq_test, X_open_test])

# Set your threshold here (try lowering it if recall is low)
threshold = 0.45

# Convert to binary predictions based on threshold
y_pred_binary = (y_pred >= threshold).astype(int).flatten()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

print("Accuracy:", accuracy_score(y_test, y_pred_binary))
print("Precision:", precision_score(y_test, y_pred_binary))
print("Recall:", recall_score(y_test, y_pred_binary))
print("F1 Score:", f1_score(y_test, y_pred_binary))

cm = confusion_matrix(y_test, y_pred_binary)
tn, fp, fn, tp = cm.ravel()
print("Confusion Matrix:")
print(f"[[TN: {tn}  FP: {fp}]]")
print(f"[[FN: {fn}  TP: {tp}]]")
print("Train set class distribution:", np.unique(y_train, return_counts=True))
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.show()
