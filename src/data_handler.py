import yfinance as yf
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from features import Features  #type: ignore
from tensorflow.keras.models import load_model #type: ignore
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


class DataHandler:
    def __init__(self, ticker, config, period=DEFAULT_PERIOD):
        self.ticker = ticker
        self.config = config
        self.period = period
        self.featurelist = ['Close', 'High', 'Low', 'Open', 'Volume', 'SMA_5', 'SMA_40', 'Volatility',
                             'Bollinger_Upper', 'Bollinger_Lower', 'Z_Score', 'RSI', 'MACD', 'MACD_Hist']
        
        self.unnormalized_features = ['Volatility', 'Bollinger_Upper', 'Bollinger_Lower', 'Z_Score', 'RSI', 'MACD', 'MACD_Hist']
        self.feature_scaler = None
        self.nextopen_scaler = None
        self.nextclose_scaler = None
        self.dates = None

    def retrieve_data(self):
        raw_data = get_data(self.ticker, self.period)
        features = Features(self.ticker, self.config['window_size'], raw_data)
        features_data = features.add_features()
        all_data = pd.concat([raw_data, features_data], axis=1)
        all_data = all_data.loc[:,~all_data.columns.duplicated()]
        all_data = all_data.dropna().reset_index()
        return all_data

    def split_datasets(self, all_data):
        train = self.config['train_split']
        val = self.config['val_split']

        train_index = int(train*len(all_data))
        val_index = int((train + val)*len(all_data))

        train_data = all_data[:train_index]
        val_data = all_data[train_index:val_index]
        test_data = all_data[val_index:]
        return train_data, val_data, test_data

    def seperate_data_bytype(self, split_dataset): #Seperates into features, next open, and nex close
        next_open = split_dataset['Open'].shift(-1)   # Open at t+1
        next_close = split_dataset['Close'].shift(-1) # Close at t+1

        all_features = split_dataset.iloc[:-1]        
        next_open = next_open[:-1]           
        next_close = next_close[:-1]

        return all_features, next_open, next_close
    
    def set_scalers(self, training_data):
        features, next_open, next_close = self.seperate_data_bytype(training_data)
        #since were only getting scalers here we dont need a "Scalers to not normalize" variable
        features_to_normalize = [f for f in self.featurelist if f not in self.unnormalized_features]
        self.features_scaler = MinMaxScaler()
        self.nextopen_scaler = MinMaxScaler()
        self.nextclose_scaler = MinMaxScaler()

        self.features_scaler.fit(features[features_to_normalize])
        self.nextopen_scaler.fit(next_open.values.reshape(-1,1))
        self.nextclose_scaler.fit(next_close.values.reshape(-1,1))

    def get_scalers(self):
        return self.features_scaler, self.nextopen_scaler, self.nextclose_scaler

    def transform(self, data, datatype='features', inverse=False):
        data = data.copy()
        datatype = datatype.lower()

        def ensure_2d(x):
            if isinstance(x, pd.Series):
                return x.values.reshape(-1, 1)
            elif isinstance(x, np.ndarray) and x.ndim == 1:
                return x.reshape(-1, 1)
            return x

        if not inverse:
            if datatype == 'features':
                features_to_normalize = [f for f in self.featurelist if f not in self.unnormalized_features]
                data[features_to_normalize] = self.features_scaler.transform(data[features_to_normalize])
                return data
            elif datatype == 'open' or datatype == 'nextopen':
                return self.nextopen_scaler.transform(ensure_2d(data))
            elif datatype == 'close' or datatype == 'nextclose':
                return self.nextclose_scaler.transform(ensure_2d(data))
            else:
                raise ValueError(f"Unknown datatype for transform {datatype}")
        else:
            if datatype == 'features':
                features_to_normalize = [f for f in self.featurelist if f not in self.unnormalized_features]
                data[features_to_normalize] = self.features_scaler.inverse_transform(data[features_to_normalize])
                return data
            elif datatype == 'open' or datatype == 'nextopen':
                return self.nextopen_scaler.inverse_transform(ensure_2d(data))
            elif datatype == 'close' or datatype == 'nextclose':
                return self.nextclose_scaler.inverse_transform(ensure_2d(data))
            else:
                raise ValueError(f"Unknown datatype for inverse transform {datatype}")

            
    def create_sequences(self, features, next_close, next_open):
        # features = np.array(features)
        # next_close = np.array(next_close)
        # next_open = np.array(next_open)
        window_size = self.config['window_size']
        X, y_close, X_open = [], [], []
        
        for i in range(len(features) - window_size):
            seq_x = features[i:i+window_size].values  # shape: (window_size, num_features)
            X.append(seq_x)
            
            # Targets are the values *immediately after* the window
            y_close.append(next_close[i + window_size])
            X_open.append(next_open[i + window_size])
            
        return np.array(X), np.array(y_close).reshape(-1, 1), np.array(X_open).reshape(-1, 1)

    def prepare_data(self, testing=False, XGBoost = False):
        all_data = self.retrieve_data()

        self.dates = all_data['Date'] if 'Date' in all_data else None    

        train_data, val_data, test_data = self.split_datasets(all_data)
        
        train_data = train_data.drop(columns=['Date', 'index'], errors='ignore')
        val_data = val_data.drop(columns=['Date', 'index'], errors='ignore')
        test_data = test_data.drop(columns=['Date', 'index'], errors='ignore')
        self.set_scalers(train_data)
        if testing:
            test_features, test_next_open, test_next_close = self.seperate_data_bytype(test_data)
            test_features = self.transform(test_features, datatype='features')
            test_next_close = self.transform(test_next_close, datatype='close')
            test_next_open = self.transform(test_next_open, datatype='open')

            if XGBoost:
                x = 1
                lstm = load_model(f'{self.ticker}_regression_model_{self.period}.h5', compile = False)

                #Get Training Data
                temp_test_features, test_next_close, test_next_open = self.create_sequences(test_features, test_next_close, test_next_open)

                test_pred_close = lstm.predict([temp_test_features, test_next_open])
                last_timestep_features = temp_test_features[:, -1, :]  
                df_features = pd.DataFrame(last_timestep_features, columns=self.featurelist)

                df_next_open = pd.Series(test_next_open.flatten(), name='next_open')
                df_pred_close = pd.Series(test_pred_close.flatten(), name='next_close').shift(-x)

                XGBoost_test_data = pd.concat([df_features, df_next_open, df_pred_close], axis=1)
                x_test = XGBoost_test_data.dropna().reset_index(drop=True)

                y_test = test_next_close[:-x]

                return x_test, y_test

            else: #else XGBoost = False
                test_features, test_next_close, test_next_open = self.create_sequences(test_features, test_next_close, test_next_open)

                return test_features, test_next_open, test_next_close
        else: #else testing
            train_features, train_next_open, train_next_close = self.seperate_data_bytype(train_data)
            train_features = self.transform(train_features, datatype='features')
            train_next_close = self.transform(train_next_close, datatype='close')
            train_next_open = self.transform(train_next_open, datatype='open')

            val_features, val_next_open, val_next_close = self.seperate_data_bytype(val_data)
            val_features = self.transform(val_features, datatype='features')
            val_next_close = self.transform(val_next_close, datatype='close')
            val_next_open = self.transform(val_next_open, datatype='open')

            val_dates = self.dates[len(train_data) : len(train_data) + len(val_data)]
            val_dates = val_dates[self.config['window_size']:]
    

            #XGBoost needs: features, open (t+1), and pred_close(t+1) all in one vector
            #NOTE: train feautures is a temp value becaue XGBoost takes in vectors, not sequcnes data s inputs, next_open is not temp because create_sequences only aligns
            #it with the features data, not sequentializes it. Since sequcentializing data chops the first n-window_size points off of the data, pred_close will also have 
            #the first n-windows_size points chopped off, and so we must chop the same number off our XGboost input uunsequentialized data also, to align with the other data points
            if XGBoost:
                lstm = load_model(f'{self.ticker}_regression_model_{self.period}.h5', compile = False)
                x = 1
                #Get Training Data
                temp_train_features, y_train, train_next_open = self.create_sequences(train_features, train_next_close, train_next_open)

                train_pred_close = lstm.predict([temp_train_features, train_next_open])
                last_timestep_features = temp_train_features[:, -1, :]  
                df_features = pd.DataFrame(last_timestep_features, columns=self.featurelist)

                df_next_open = pd.Series(train_next_open.flatten(), name='next_open')
                df_pred_close = pd.Series(train_pred_close.flatten(), name='next_close').shift(-x)

                x_train = pd.concat([df_features, df_next_open, df_pred_close], axis=1)
                x_train = x_train.dropna().reset_index(drop=True)

                #Get Val Data:
                temp_val_features, y_val, val_next_open = self.create_sequences(val_features, val_next_close, val_next_open)
                val_pred_close = lstm.predict([temp_val_features, val_next_open])
                val_last_timestep_features = temp_val_features[:, -1, :]  
                val_df_features = pd.DataFrame(val_last_timestep_features, columns=self.featurelist)

                val_df_next_open = pd.Series(val_next_open.flatten(), name='next_open')
                val_df_pred_close = pd.Series(val_pred_close.flatten(), name='next_close').shift(-x)

                x_val = pd.concat([val_df_features, val_df_next_open, val_df_pred_close], axis=1)
                x_val = x_val.dropna().reset_index(drop=True)


                y_train = y_train[:-x]

                y_val = y_val[:-x]


                return x_train, x_val, y_train, y_val

            else:

                train_features, train_next_close, train_next_open = self.create_sequences(train_features, train_next_close, train_next_open)
                val_features, val_next_close, val_next_open = self.create_sequences(val_features, val_next_close, val_next_open)

                return train_features, train_next_open, train_next_close, val_features, val_next_open, val_next_close, val_dates


