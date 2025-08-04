class Features:
    def __init__(self, ticker, window, df):
        self.ticker = ticker
        self.window = window
        self.df = df.copy()
        self.price = self.df['Close']  # Use Close price for feature calc

        self.residual = None  # For now, could be set to zero or residual from another model if you have one

    def calc_volatility(self):
        returns = self.price.pct_change()
        self.df['Volatility'] = returns.rolling(self.window).std()

    def calc_bollinger_bands(self):
        sma = self.price.rolling(self.window).mean()
        rolling_std = self.price.rolling(self.window).std()
        self.df['Bollinger_Upper'] = sma + 2 * rolling_std
        self.df['Bollinger_Lower'] = sma - 2 * rolling_std

    def calc_z_score(self):
        # Example z-score of price vs rolling mean
        rolling_mean = self.price.rolling(self.window).mean()
        rolling_std = self.price.rolling(self.window).std()
        self.df['Z_Score'] = (self.price - rolling_mean) / rolling_std

    def add_all_features(self):
        self.calc_volatility()
        self.calc_bollinger_bands()
        self.calc_z_score()
        return self.df