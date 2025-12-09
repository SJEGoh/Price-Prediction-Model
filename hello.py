import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import t

# -------------------------
# 1. Load data
# -------------------------
ticker = "BTC-USD"
df = yf.Ticker(ticker).history(period="5y")
prices = df["Close"]
returns = prices.pct_change().dropna()

# Train / test split
split = int(len(returns) * 0.8)
train_returns = returns[:split]
test_returns  = returns[split:]

sd = train_returns.std()
mu = train_returns.mean()

P0 = prices.iloc[split]  # starting price for test window

# Actual price path over test window
actual_prices = prices.iloc[split+1:]              # skip the NaN from pct_change
actual_prices = actual_prices.iloc[:len(test_returns)]  # align with test_returns
test_index = actual_prices.index
n_steps = len(test_returns)

# -------------------------
# 2. Monte Carlo simulation
# -------------------------
n_paths = 10000

# Z ~ N(0,1) for all paths and steps
Z = np.random.normal(size=(n_paths, n_steps))

# Forecast returns for all paths: r = mu + sd * Z
forecast_returns = mu + sd * Z                     # shape (n_paths, n_steps)

# Convert returns to price paths: P_t = P0 * Π (1 + r_t)
pred_prices = P0 * (1 + forecast_returns).cumprod(axis=1)   # shape (n_paths, n_steps)

# -------------------------
# 3. Find "most likely" (central) path
# -------------------------
terminal_prices = pred_prices[:, -1]               # final price for each path
median_terminal = np.median(terminal_prices)

# Index of path whose final price is closest to the median
idx_central = np.argmin(np.abs(terminal_prices - median_terminal))

central_prices = pred_prices[idx_central]         
central_returns = forecast_returns[idx_central]    

mean_daily = central_returns.mean()
std_daily = central_returns.std(ddof=1)

trading_days = 365  # BTC trades every day; use 252 for equities
sharpe_annual = (mean_daily / std_daily) * np.sqrt(trading_days)

print(f"Annualised Sharpe of 'most likely' path: {sharpe_annual:.3f}")

plt.figure(figsize=(12, 6))

plt.plot(
    test_index, central_prices,
    label=f"Most likely MC path (Sharpe ≈ {sharpe_annual:.2f})",
    color='blue', linewidth=2
)

plt.plot(
    test_index, actual_prices.values,
    label="Actual Price",
    color='black', linewidth=2
)

plt.title(f"{ticker} - Monte Carlo Price Paths\nMost likely path + Annualised Sharpe")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
