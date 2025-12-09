from hmmlearn.hmm import GaussianHMM
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import t

class HmmDetect:
    def __init__(self, logreturns_data, means = 3):
        self.model = GaussianHMM(n_components=means, covariance_type="full", n_iter=1000, verbose = True)
        self.model.fit(logreturns_data.reshape(-1, 1))

        _, posteriors = self.model.score_samples(logreturns_data.reshape(-1, 1))

        self.mu = self.model.means_.flatten()
        self.sigma = np.sqrt(self.model.covars_).flatten()
        self.P = self.model.transmat_
        self.pi_last = posteriors[-1]
        self.df = np.zeros(self.K)
    
    def sim_reg_path(self, n_steps):
        K = len(self.pi_last)
        S = np.zeros(n_steps, dtype = int)

        S[0] = np.random.choice(K, p = self.pi_last)

        for t in range(1, n_steps):
            S[t] = np.random.choice(K, p = self.P[S[t-1]])
        return S
    
    def sim_returns(self, S):
        nu = self.df
        t_raw = t.rvs(df=nu, size=len(S))
        t_std = t_raw / np.sqrt(nu / (nu - 2.0))
        return self.mu[S] + self.sigma[S] * t_std
    
    def returns_to_prices(self, P0, r):
        return P0 * np.exp(r.cumsum())


def main():
    ticker = "NVDA"
    df = yf.Ticker(ticker).history(period="1y")
    prices = df["Close"]
    logreturns = np.log(prices/prices.shift(1)).dropna().values.reshape(-1, 1)

    split = int(len(logreturns) * 0.8)
    train_returns = logreturns[:split]
    test_returns  = logreturns[split:] 
    detector = HmmDetect(train_returns, 3)

    n_steps = len(test_returns)
    n_paths = 5000
    P0 = prices.iloc[split]

    price_paths = np.zeros((n_paths, n_steps))
    for i in range(n_paths):
        S = detector.sim_reg_path(n_steps)
        r = detector.sim_returns(S)
        price_paths[i] = detector.returns_to_prices(P0, r)
    median_curve = np.median(price_paths, axis=0)   # (n_steps,)

# squared L2 distance of each path to the median curve
    dists = np.sum((price_paths - median_curve)**2, axis=1)
    idx = np.argmin(dists)
    central_path = price_paths[idx]

    central_returns = np.diff(central_path) / central_path[:-1]
    mean_daily = central_returns.mean()
    std_daily  = central_returns.std()
    sharpe = (mean_daily / std_daily) * np.sqrt(252)
    print("Estimated Sharpe:", sharpe)
    actual_prices = prices.iloc[split : split+1+n_steps]

    central_path = np.insert(central_path, 0, P0, axis = 0)
    plt.figure(figsize=(12,6))
    plt.plot(actual_prices.index, actual_prices, label="Actual")
    plt.plot(actual_prices.index, central_path, color='blue', 
            label=f"Most Likely Path (Sharpe={sharpe:.2f})")

    plt.legend()
    plt.grid(True)
    plt.show()




if __name__ == "__main__":
    main()


