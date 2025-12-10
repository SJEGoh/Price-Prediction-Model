import pandas as pd
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

ticker = "BTC-USD"
df = yf.Ticker(ticker).history(period="1y")
prices = df["Close"]
returns = np.log(prices/prices.shift(1)).dropna().values.reshape(-1, 1)

split = int(len(returns) * 0.8)
train_returns = returns[:split]
test_returns  = returns[split:]

model = GaussianHMM(n_components=3, covariance_type="full", n_iter=500)
model.fit(train_returns.reshape(-1, 1))

train_logprob, train_posteriors = model.score_samples(train_returns.reshape(-1, 1))

mu = model.means_.flatten()
sigma = np.sqrt(model.covars_.flatten())
P = model.transmat_
pi_last = train_posteriors[-1]

def simulate_regime_path(P, pi0, n_steps):
    K = len(pi0)
    S = np.zeros(n_steps, dtype=int)
    
    # pick first regime using last posterior of training
    S[0] = np.random.choice(K, p=pi0)
    
    for t in range(1, n_steps):
        S[t] = np.random.choice(K, p=P[S[t-1]])
    
    return S
print(simulate_regime_path(P, pi_last, 100))


def simulate_returns(S, mus, sigmas):
    return np.random.normal(loc=mus[S], scale=sigmas[S])

n_steps = len(test_returns)
n_paths = 5000
P0 = prices.iloc[split]

def returns_to_prices(r, P0):
    return P0 * np.cumprod(1 + r)
price_paths = np.zeros((n_paths, n_steps))

for i in range(n_paths):
    S = simulate_regime_path(P, pi_last, n_steps)
    r = simulate_returns(S, mu, sigma)
    price_paths[i] = returns_to_prices(r, P0)

terminal = price_paths[:, -1]
median_terminal = np.median(terminal)
idx_central = np.argmin(np.abs(terminal - median_terminal))

central_path = price_paths[idx_central]
central_returns = np.diff(central_path) / central_path[:-1]

mean_daily = central_returns.mean()
std_daily  = central_returns.std()
sharpe = (mean_daily / std_daily) * np.sqrt(365)
print("Sharpe:", sharpe)

actual_prices = prices.iloc[split+1 : split+1+n_steps]

plt.figure(figsize=(12,6))
plt.plot(actual_prices.index, actual_prices, label="Actual")

plt.plot(actual_prices.index, central_path, color='blue', 
         label=f"Most Likely Path (Sharpe={sharpe:.2f})")

plt.fill_between(actual_prices.index,
                 np.percentile(price_paths, 5, axis=0),
                 np.percentile(price_paths, 95, axis=0),
                 alpha=0.3, label="5–95% range")

plt.legend()
plt.grid(True)
plt.show()

