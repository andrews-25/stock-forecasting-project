import yfinance as yf
import pandas as pd
import os

SAVE_PATH = "data/processed"


def download_data(ticker, start_date):
    df = yf.download(ticker, start=start_date)

    os.makefdirs(SAVE_PATH, exist_ok=True)


    processed_path = os.path.join(SAVE_PATH, f"{ticker}.pkl")
    df.to_pickle(processed_path)

    print(f"Data for {ticker} downloaded and saved to {processed_path}")
    return df

def load_data(ticker):
    processed_path = os.path.join(SAVE_PATH, f"{ticker}.pkl")
    if os.path.exists(processed_path):
        return pd.read_pickle(processed_path)
    else:
        print(f"No processed data found for {ticker}. Please download it first")
        return None