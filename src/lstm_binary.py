import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import tensorflow as tf
from tensorflow.keras.models import Model            # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate # type: ignore
from tensorflow.keras.optimizers import Adam                # type: ignore
from tensorflow.keras.losses import BinaryCrossentropy                    # type: ignore
from tensorflow.keras.regularizers import l2  #type: ignore
from keras_cv.losses import FocalLoss  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from data_handler import LSTMDataHandler
import matplotlib.pyplot as plt

def find_best_threshold(y_true, y_pred_probs, thresholds=np.arange(0, 1.01, 0.01)):
    best_thresh = 0.5
    best_f1 = 0
    for thresh in thresholds:
        preds = (y_pred_probs >= thresh).astype(int).flatten()
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


config = {
    #Data
    'seed': 100,
    'train_split': 0.8,            
    'window_size': 30,            
    #NMetwork
    'lstm_units_1': 60,
    'lstm_units_2': 30,
    'dense_units_open': 20,
    'merge_dense_units': 16,
    #Dropout
    'dropout_rate_1': 0.2,
    'dropout_rate_2': 0.2,
    'merge_dropout_rate': 0.2,
    #Training
    'learning_rate': 0.001,
    'batch_size': 10,
    'epochs': 1000,
    #Callbacks
    'early_stopping_patience': 30,
    'early_stopping_min_delta': 1e-7,
    'reduce_lr_factor': 0.65,
    'reduce_lr_patience': 15,
    'reduce_lr_min_lr': 1e-6,
    #Classification
    'regularization_strength': 0.0001,
    #'threshold': 0.3
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

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)



bce_loss = BinaryCrossentropy()
focal_loss = FocalLoss(alpha = .4, gamma = 2)
def weighted_loss(y_true, y_pred):
    bce = bce_loss(y_true, y_pred)
    focal = focal_loss(y_true, y_pred)
    return .8*bce + .2*focal

model.compile(
    optimizer=Adam(learning_rate=config['learning_rate']),
    loss= weighted_loss,
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
    verbose=1,
    #class_weight = {}
)


### MODEL EVALUATION ###

# Predict probabilities (assuming model outputs shape (n_samples, 1))
y_pred = model.predict([X_seq_test, X_open_test])

best_threshold, best_f1 = find_best_threshold(y_test, y_pred)
print(f"Best threshold found: {best_threshold:.2f} with F1 score: {best_f1:.4f}")

# Convert to binary predictions based on threshold
y_pred_binary = (y_pred >= best_threshold).astype(int).flatten()

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
print("Test set class distribution:", np.unique(y_test, return_counts=True))



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
