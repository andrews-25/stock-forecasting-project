import yfinance as yf
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from features import Features

SAVE_PATH = "data/processed"
DEFAULT_PERIOD = "5y"


def download_data(ticker, period=DEFAULT_PERIOD):
    df = yf.download(ticker, period=period)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    os.makedirs(SAVE_PATH, exist_ok=True)
    processed_path = os.path.join(SAVE_PATH, f"{ticker}_{period}.pkl")
    df.to_pickle(processed_path)
    print(f"Data for {ticker} downloaded and saved to {processed_path}")
    return df


def get_data(ticker, period=DEFAULT_PERIOD):
    processed_path = os.path.join(SAVE_PATH, f"{ticker}_{period}.pkl")
    if os.path.exists(processed_path):
        return pd.read_pickle(processed_path)
    else:
        download_data(ticker, period)
        return pd.read_pickle(processed_path)


class LSTMDataHandler:
    def __init__(self, ticker, config, period=DEFAULT_PERIOD, target_type='regression'):
        self.target_type = target_type
        self.ticker = ticker
        self.config = config
        self.period = period

        # Load raw data
        raw_df = get_data(ticker, period)

        # Compute all features
        features_obj = Features(ticker, config['window_size'], raw_df)
        features_obj.add_normalized_features()
        features_obj.add_unnormalized_features()
        self.df = features_obj.df.dropna().reset_index(drop=True)

        # Initialize scaler
        self.scaler = MinMaxScaler()

        # Define normalized and unnormalized feature sets
        self.normalized_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_40']
        self.unnormalized_features = ['Volatility','Bollinger_Upper','Bollinger_Lower',
                                      'Z_Score','RSI','MACD','MACD_Hist']

        # Combine for sequence creation
        self.features = self.normalized_features + self.unnormalized_features

    def normalize(self):
        # Split training set for fitting scaler
        split_index = int(len(self.df) * self.config['train_split'])
        train_df = self.df.iloc[:split_index]

        # Fit scaler on normalized features
        self.scaler.fit(train_df[self.normalized_features])

        # Scale normalized features
        scaled_data = self.scaler.transform(self.df[self.normalized_features])
        scaled_df = pd.DataFrame(scaled_data, columns=self.normalized_features, index=self.df.index)

        # Concatenate with unnormalized features
        final_df = pd.concat([scaled_df, self.df[self.unnormalized_features]], axis=1)

        # Ensure correct column order
        final_df = final_df[self.features]

        return final_df

    def create_sequences(self, normalized_df):
        window = self.config['window_size']
        input_seq = []
        input_open = []
        target_list = []

        for i in range(window + 1, len(normalized_df) - 1):
            seq = normalized_df[self.features].iloc[i - window:i].values
            next_open = normalized_df.iloc[i + 1]['Open']
            next_close = normalized_df.iloc[i + 1]['Close']

            input_seq.append(seq)
            input_open.append([next_open])

            if self.target_type == 'regression':
                target = next_close
            elif self.target_type == 'classification':
                target = 1 if next_close > next_open else 0
            else:
                raise ValueError("Invalid target_type. Use 'regression' or 'classification'.")

            target_list.append(target)

        return np.array(input_seq), np.array(input_open), np.array(target_list)

    def prepare_data(self, normalize=True, training=True):
        # Normalize if requested
        df = self.normalize() if normalize else self.df[self.features].copy()

        # Create sequences
        X_seq, X_open, y = self.create_sequences(df)

        # Split into train/val/test
        train_split_index = int(len(X_seq) * self.config['train_split'])
        val_split_index = int(len(X_seq) * (self.config['train_split'] + self.config['val_split']))
        test_split_index = len(X_seq)

        X_seq_train = X_seq[:train_split_index]
        X_open_train = X_open[:train_split_index]
        y_train = y[:train_split_index]

        X_seq_val = X_seq[train_split_index:val_split_index]
        X_open_val = X_open[train_split_index:val_split_index]
        y_val = y[train_split_index:val_split_index]

        X_seq_test = X_seq[val_split_index:test_split_index]
        X_open_test = X_open[val_split_index:test_split_index]
        y_test = y[val_split_index:test_split_index]

        if training:
            return (X_seq_train, X_seq_val, X_open_train, X_open_val, y_train, y_val), self.scaler
        else:
            return (X_seq_test, X_open_test, y_test), self.scaler
