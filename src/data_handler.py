import yfinance as yf
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
import os
import numpy as np
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
        raw_df = get_data(ticker, period)
        self.olhcv_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        features = Features(ticker, config['window_size'], raw_df)
        self.df = features.add_all_features().dropna().reset_index(drop=True)   
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Volatility','Bollinger_Upper', 'Bollinger_Lower', 'Z_Score']
        self.scaler = MinMaxScaler()
    def normalize(self):
        split_index = int(len(self.df) * self.config['train_split'])
        train_df = self.df.iloc[:split_index]

      
        olhcv_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        engineered_features = [f for f in self.features if f not in olhcv_features]
        
      
        self.scaler.fit(train_df[olhcv_features])

      
        scaled_olhcv = self.scaler.transform(self.df[olhcv_features])
        scaled_olhcv_df = pd.DataFrame(scaled_olhcv, columns=olhcv_features, index=self.df.index)

  
        normalized_df = pd.concat([scaled_olhcv_df, self.df[engineered_features]], axis=1)
        #print("Columns after concatenation:", normalized_df.columns.tolist())

        # Ensure column order matches self.features list
        normalized_df = normalized_df[self.features]

        return normalized_df

    def create_sequences(self, normalized_df):
        window = self.config['window_size']
        input_seq = []
        input_open = []
        target_list = []

        for i in range(window + 1, len(normalized_df)):
            seq = normalized_df[self.features].iloc[(i - 1) - window:(i - 1)].values
            current_open = normalized_df.iloc[i]['Open']
            current_close = normalized_df.iloc[i]['Close']

            input_seq.append(seq)
            input_open.append([current_open])

            if self.target_type == 'regression':
                target = current_close
            elif self.target_type == 'classification':
                target = 1 if current_close > current_open else 0
            else:
                raise ValueError("Invalid target_type. Use 'regression' or 'classification'.")

            target_list.append(target)

        return np.array(input_seq), np.array(input_open), np.array(target_list)

    def prepare_data(self, normalize=True, training=True):
        if normalize:
            df = self.normalize()
        else:
            df = self.df[self.features].copy()
        X_seq, X_open, y = self.create_sequences(df)

        #get train split index
        train_split_index = int(len(X_seq) * self.config['train_split'])
        X_seq_train = X_seq[:train_split_index]
        X_open_train = X_open[:train_split_index]
        y_train = y[:train_split_index]
        #get validation split index
        val_split_index = int(len(X_seq) * (self.config['train_split'] + self.config['val_split']))
        X_seq_val = X_seq[train_split_index:val_split_index]
        X_open_val = X_open[train_split_index:val_split_index]
        y_val = y[train_split_index:val_split_index]
        #get test split index
        test_split_index = int(len(X_seq))
        X_seq_test = X_seq[val_split_index:test_split_index]
        X_open_test = X_open[val_split_index:test_split_index]
        y_test = y[val_split_index:test_split_index]

        if training:
            return (X_seq_train, X_seq_val, X_open_train, X_open_val, y_train, y_val), self.scaler
        else:
            return (X_seq_test, X_open_test, y_test), self.scaler

