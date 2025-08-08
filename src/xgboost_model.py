from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from data_handler import DataHandler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
lstmconfig = {
    'seed': 100,
    'train_split': 0.7,
    'val_split': .15,
    'test_split': .15,
    'window_size': 20,
    'lstm_units_1': 128,
    'lstm_units_2': 64,
    'lstm_units_3': 32,
    'dense_units_open': 32, 
    'dense_units_concat': 16,
    'dropout_rate_1': 0.25,
    'dropout_rate_2': 0.25,
    'dropout_rate_3': 0.25,
    'dropout_rate_final': 0.25,
    'learning_rate': 0.008,
    'batch_size': 8,
    'epochs': 1000,
    'regularization_strength': 5e-4,
    'early_stopping_patience': 30,
    'early_stopping_min_delta': 0.00002,
    'reduce_lr_factor': 0.5,
    'reduce_lr_patience': 10,
    'reduce_lr_min_lr': 1e-6,
}

params = {
    'objective': 'reg:squarederror',  # Loss function for regression problems
    'eval_metric': 'rmse',            # Metric to monitor during training
    'max_depth': 4,                   # Maximum tree depth (limits complexity)
    'eta': 0.05,                     # Learning rate (step size for updates)
    'subsample': 0.8,                # Fraction of samples used per tree
    'colsample_bytree': 0.8,          # Fraction of features used per tree
    'min_child_weight': 3,            # Minimum samples required in leaf nodes
    'gamma': 1,                       # Minimum loss reduction to make split
    'alpha': .1,                       # L1 regularization strength
    'lambda': 3,                      # L2 regularization strength
    'seed': 100                       # Random seed for reproducibility
}

evals_result = {}

ticker = input("Ticker: ").upper()
data_handler = DataHandler(ticker, lstmconfig)

x_train, x_val, y_train, y_val = data_handler.prepare_data(XGBoost=True)

# Make sure y_train and y_val are 1D arrays of continuous target prices (no binary labels)
y_train = y_train.flatten()
y_val = y_val.flatten()




dtrain = xgb.DMatrix(x_train, label=y_train)
dval = xgb.DMatrix(x_val, label=y_val)

evals = [(dtrain, 'train'), (dval, 'validation')]

model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=5000,
    evals=evals,
    early_stopping_rounds=25,
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
