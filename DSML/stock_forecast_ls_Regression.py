import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ==========================================
# PART 1: GENERATE CSV (Run once)
# We'll generate a simple time series of closing prices and date index
# ==========================================
days = 200
dates = pd.date_range(end=pd.Timestamp.today(), periods=days).strftime('%Y-%m-%d')
# create a synthetic trend + seasonality + noise
t = np.arange(days)
prices = 100 + 0.05*t + 2*np.sin(2*np.pi*t/30) + np.random.normal(0,1.5,days)
df_gen = pd.DataFrame({'date': dates, 'close_price': np.round(prices, 2)})
df_gen.to_csv('stock_prices.csv', index=False)
print("✅ 'stock_prices.csv' generated. Columns:", list(df_gen.columns))


# ==========================================
# PART 2: SOLUTION (Uncomment to run after CSV generated)
# Use simple least squares (polynomial fit) on historical close prices to forecast next values.
# ==========================================
'''
# 1. Load
df = pd.read_csv('stock_prices.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# 2. Create numeric time index
df['t'] = np.arange(len(df))

# 3. Fit least squares (degree 2 polynomial)
coeffs = np.polyfit(df['t'], df['close_price'], deg=2)
poly = np.poly1d(coeffs)

# 4. Predict on existing and forecast next 7 days
df['pred'] = poly(df['t'])
future_t = np.arange(len(df), len(df)+7)
future_preds = poly(future_t)

# 5. Evaluate on last 20% as simple holdout
train_n = int(0.8*len(df))
mse = mean_squared_error(df['close_price'].iloc[train_n:], df['pred'].iloc[train_n:])
print("\\n--- Stock Forecast (Least Squares) ---")
print(f"MSE on holdout: {mse:.4f}")

# 6. Plot
plt.figure(figsize=(8,4))
plt.plot(df['date'], df['close_price'], label='Actual')
plt.plot(df['date'], df['pred'], label='Polynomial Fit')
future_dates = pd.date_range(df['date'].iloc[-1] + pd.Timedelta(days=1), periods=7)
plt.plot(future_dates, future_preds, '--', label='Forecast')
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("Stock Price and Least Squares Forecast")
plt.legend()
plt.tight_layout()
plt.show()
'''
