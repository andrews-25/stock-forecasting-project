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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from data_handler import get_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
seed = 100
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'  # For more determinism on GPU




#Device Check
print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("Using device: GPU")
else:
    print("Using device: CPU")
    
### Get Data and Normalize ###
def normalize(df, features):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    scaled_df = pd.DataFrame(scaled, columns = features, index = df.index)
    return  scaled_df, scaler 
ticker = input("Enter Ticker: ").upper()
df = get_data(ticker)
features = ['Open', 'High', 'Low', 'Close', 'Volume']
normalized_df, scaler = normalize(df, features)

window = 20
input_seq = [] #LHCV inputs
input_open = [] #current open
target_close = [] #Todays Close (Target)

#Fill input and target sequences
for i in range (window + 1, len(df)):
    sequence = normalized_df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[(i-1)-window:(i-1)].values
    current_open = normalized_df.iloc[i]['Open']
    current_close = normalized_df.iloc[i]['Close']
    
    input_seq.append(sequence)
    input_open.append([current_open])
    target_close.append(current_close)


#Data Prep
X_seq = np.array(input_seq)
X_open = np.array(input_open)
y = np.array(target_close)
X_seq_train, X_seq_test, X_open_train, X_open_test, y_train, y_test = train_test_split(X_seq, X_open, y, test_size = 0.2, shuffle = False)

lstm_input = Input(shape=(window, 5), name = 'lstm_input')
lstm_output = LSTM(128, return_sequences = False, name = 'lstm_layer')(lstm_input)
lstm_output_dropout = Dropout(0.3, name = 'lstm_dropout')(lstm_output)

model_input_open = Input(shape=(1,), name = 'model_input_open')
dense_open = Dense(32, activation = 'relu', name='open_dense')(model_input_open)
concat = Concatenate(name = 'concat')([lstm_output_dropout, dense_open])

model_dense_1 = Dense(128, activation='relu', name='dense_1')(concat)
model_dropout_2 = Dropout(0.2, name = 'model_dropout')(model_dense_1)
model_output = Dense(1, name = 'model_output')(model_dropout_2)

model = Model(inputs = [lstm_input, model_input_open], outputs = model_output, name = 'ClosePredictorLSTM')
opt = Adam(learning_rate=.001)
model.compile(opt, loss='mse')

history =  model.fit(
    [X_seq_train, X_open_train],
    y_train,
    validation_data=([X_seq_test, X_open_test], y_test),
    epochs = 100,
    batch_size=8,
    verbose = 1
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

#Print a few comparisons
print("Sample predictions vs actual (denormalized):")
for i in range(5):
    print(f"Predicted: ${predicted_close_real[i]:.2f} | Actual: ${y_test_real[i]:.2f}")

#Run stats
mse = mean_squared_error(y_test_real, predicted_close_real)
mae = mean_absolute_error(y_test_real, predicted_close_real)
r2 = r2_score(y_test_real, predicted_close_real)

print(f"\n Model Performance:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()
