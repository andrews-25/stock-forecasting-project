import yfinance as yf
import pandas as pd
from datetime import date
from dateutil.relativedata import relativedelta
import os

SAVE_PATH = "data/processed"
DEFAULT_START = date.today() - relativedelta(years=3)


def download_data(ticker, start_date = DEFAULT_START):
    df = yf.download(ticker, start=start_date)

    os.makedirs(SAVE_PATH, exist_ok=True)


    processed_path = os.path.join(SAVE_PATH, f"{ticker}.pkl")
    df.to_pickle(processed_path)

    print(f"Data for {ticker} downloaded and saved to {processed_path}")
    return df

def get_data(ticker, start_date = DEFAULT_START):
    processed_path = os.path.join(SAVE_PATH, f"{ticker}.pkl")
    if os.path.exists(processed_path):
        return pd.read_pickle(processed_path)
    else:
        download_data(ticker, start_date)
        return pd.read_pickle(processed_path)
