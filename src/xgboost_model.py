from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from data_handler import DataHandler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import shap
from lstm_regression import config as lstmconfig


params = {
    #'device': 'cuda',
    'objective': 'reg:squarederror',  # Loss function for regression problems
    #'objective': 'reg:quantileerror',
    #'quantile_alpha': 0.1,  # median regression

    'eval_metric': 'rmse',      # Metric to monitor during training
    'tree_method': 'exact',    # or 'gpu_hist' if using GPU
    #'max_leves': 2,   
                      
    'max_depth': 10,                   # Maximum tree depth (limits complexity)
    'eta': 0.003,                     # Learning rate (step size for updates)
    'subsample': .8,                # Fraction of samples used per tree
    'colsample_bytree': .8,          # Fraction of features used per tree
    #'min_child_weight': 2,            # Minimum samples required in leaf nodes
    'gamma': 0,                       # Minimum loss reduction to make split
    'alpha': 0,                       # L1 regularization strength
    'lambda': 5,                      # L2 regularization strength
    'seed': 100                       # Random seed for reproducibility
}
import sys
evals_result = {}

ticker = input("Ticker: ").upper()
data_handler = DataHandler(ticker, lstmconfig, period = lstmconfig['period'])

x_train, x_val, y_train, y_val = data_handler.prepare_data(XGBoost=True)



# Make sure y_train and y_val are 1D arrays of continuous target prices (no binary labels)
y_train = y_train.flatten()
y_val = y_val.flatten()


def test_data(x_train, y_train, x_val, y_val):
    #df.drop(columns=['column_name'], inplace=True)
    included_features = [ 'Volatility']
    excluded_features = ['Open', 'Low', 'Close', 'SMA_5', 'Bollinger_Upper', 'MACD_Hist', 'High', 'RSI', 'Z_Score', 'SMA_40', 'Bollinger_Lower', 'MACD', 'Volume']
    x_train.drop(columns = excluded_features, inplace=True)
    x_val.drop(columns = excluded_features, inplace=True)
    # x_train.drop(columns = ['next_open'], inplace=True)
    # x_val.drop(columns = ['next_open' ], inplace=True)

    x_train['t+2_close'] = x_train['next_close'].shift(-1)
    x_train = x_train.dropna(subset=['t+2_close']).reset_index(drop=True)

    x_val['t+2_close'] = x_val['next_close'].shift(-1)
    x_val = x_val.dropna(subset=['t+2_close']).reset_index(drop=True)

    x_train['pred_2day_spread'] = x_train['t+2_close'] - x_train['next_open']
    x_val['pred_2day_spread'] = x_val['t+2_close'] - x_val['next_open']

    x_train['t+3_close'] = x_train['t+2_close'].shift(-1)
    x_train = x_train.dropna(subset=['t+3_close']).reset_index(drop=True)

    x_val['t+3_close'] = x_val['t+2_close'].shift(-1)
    x_val = x_val.dropna(subset=['t+3_close']).reset_index(drop=True)

    y_train = y_train[:len(x_train)]
    y_val = y_val[:len(x_val)]

    x_train['pred_spread'] = x_train['next_close'] - x_train['next_open']
    x_val['pred_spread'] = x_val['next_close'] - x_val['next_open']
    return x_train, y_train, x_val, y_val


new_xtrain, new_ytrain, new_xval, new_yval = test_data(x_train, y_train, x_val, y_val)
print(new_xtrain.head())
sys.exit()

dtrain = xgb.DMatrix(x_train, y_train)
dval = xgb.DMatrix(x_val, y_val)

evals = [(dtrain, 'train'), (dval, 'validation')]



model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=50000,
    evals=evals,
    early_stopping_rounds=100,
    verbose_eval=True,
    evals_result=evals_result
)


# Predict continuous values
y_pred = model.predict(dval)

y_pred = data_handler.transform(y_pred, datatype='close', inverse=True)
y_val = data_handler.transform(y_val, datatype='close', inverse=True)
y_val = y_val.flatten()
y_pred = y_pred.flatten()

# Evaluate regression metrics
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("Sample predictions vs actual:")
for i in range(min(5, len(y_pred))):
    print(f"Predicted: ${y_pred[i]:.2f} | Actual: ${y_val[i]:.2f}")

print(f"\nModel Performance:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R Squared Score (R²): {r2:.4f}")

# Plot training and validation RMSE over boosting rounds
train_rmse = evals_result['train']['rmse']
val_rmse = evals_result['validation']['rmse']

plt.figure(figsize=(10, 6))
plt.plot(train_rmse, label='Training RMSE')
plt.plot(val_rmse, label='Validation RMSE')
plt.xlabel('Boosting Round')
plt.ylabel('RMSE')
plt.title('Training and Validation RMSE over Boosting Rounds')
plt.legend()
plt.grid(True)
plt.show()

# Plot actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.plot(y_val, label='Actual Close Price', color='blue')
plt.plot(y_pred, label='Predicted Close Price', color='orange')
plt.title('Actual vs Predicted Close Price')
plt.xlabel('Sample Index')
plt.ylabel('Close Price')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
print(model.get_score(importance_type='weight')
)


