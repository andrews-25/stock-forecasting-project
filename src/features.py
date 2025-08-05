import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands

class Features:
    def __init__(self, ticker, window, df):
        self.ticker = ticker
        self.window = window
        self.df = df.copy()
        self.close = self.df['Close']  # Use Close close for feature calc

        self.residual = None  # For now, could be set to zero or residual from another model if you have one

    def calc_volatility(self):
        returns = self.close.pct_change()
        self.df['Volatility'] = returns.rolling(self.window).std()

    def calc_bollinger_bands(self):
        sma = self.close.rolling(self.window).mean()
        rolling_std = self.close.rolling(self.window).std()
        self.df['Bollinger_Upper'] = sma + 2 * rolling_std
        self.df['Bollinger_Lower'] = sma - 2 * rolling_std

    def calc_z_score(self):
        # Example z-score of close vs rolling mean
        rolling_mean = self.close.rolling(self.window).mean()
        rolling_std = self.close.rolling(self.window).std()
        self.df['Z_Score'] = (self.close - rolling_mean) / rolling_std

    def add_all_features(self):
        self.calc_volatility()
        self.calc_bollinger_bands()
        self.calc_z_score()
        return self.df
    