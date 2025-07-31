# === Imports ===
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential             # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
import matplotlib.pyplot as plt
from data_handler import load_data



#Device Check
print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("Using device: GPU")
else:
    print("Using device: CPU")

