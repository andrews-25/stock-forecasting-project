import yfinance as yf
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

SAVE_PATH = "data/processed"
DEFAULT_START = date.today() - relativedelta(years=3)


def download_data(ticker, start_date=DEFAULT_START):
    df = yf.download(ticker, start=start_date)
    os.makedirs(SAVE_PATH, exist_ok=True)

    processed_path = os.path.join(SAVE_PATH, f"{ticker}.pkl")
    df.to_pickle(processed_path)

    print(f"Data for {ticker} downloaded and saved to {processed_path}")
    return df


def get_data(ticker, start_date=DEFAULT_START):
    processed_path = os.path.join(SAVE_PATH, f"{ticker}.pkl")
    if os.path.exists(processed_path):
        return pd.read_pickle(processed_path)
    else:
        download_data(ticker, start_date)
        return pd.read_pickle(processed_path)



class LSTMDataHandler:
    def __init__(self, ticker, config, start_date=DEFAULT_START, target_type='regression'):
        self.target_type = target_type
        self.ticker = ticker
        self.config = config
        self.start_date = start_date
        self.df = get_data(ticker, start_date)
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.scaler = MinMaxScaler()

    def normalize(self):

        split_index = int(len(self.df) * self.config['train_split'])
        train_df = self.df.iloc[:split_index]

        self.scaler.fit(train_df[self.features])
        scaled = self.scaler.transform(self.df[self.features])
        normalized_df = pd.DataFrame(scaled, columns=self.features, index=self.df.index)
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

    def prepare_data(self, normalize=True):
        if normalize:
            df = self.normalize()
        else:
            df = self.df[self.features].copy()
        X_seq, X_open, y = self.create_sequences(df)

        split_index = int(len(X_seq) * self.config['train_split'])

        X_seq_train = X_seq[:split_index]
        X_seq_test = X_seq[split_index:]

        X_open_train = X_open[:split_index]
        X_open_test = X_open[split_index:]

        y_train = y[:split_index]
        y_test = y[split_index:]

        return (X_seq_train, X_seq_test, X_open_train, X_open_test, y_train, y_test), self.scaler
    
