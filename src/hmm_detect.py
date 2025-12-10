from hmmlearn.hmm import GaussianHMM
from options import calibrate_heston
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from ML import MLDrift
import pandas as pd

class HmmDetect:
    def __init__(self, logreturns_data, means = 3):
        self.model = GaussianHMM(n_components=means, covariance_type="full", n_iter=1000, verbose = True)
        self.model.fit(logreturns_data.reshape(-1, 1))

        _, posteriors = self.model.score_samples(logreturns_data.reshape(-1, 1))

        self.mu = self.model.means_.flatten()
        self.sigma = np.sqrt(self.model.covars_).flatten()
        self.P = self.model.transmat_
        self.pi_last = posteriors[-1] # self.df = np.zeros(self.K)

    def returns_to_prices(self, P0, r):
        return P0 * np.exp(r.cumsum())
        
    def sim_reg_path(self, n_steps):
        K = len(self.pi_last)
        S = np.zeros(n_steps, dtype = int)

        S[0] = np.random.choice(K, p = self.pi_last)

        for t in range(1, n_steps):
            S[t] = np.random.choice(K, p = self.P[S[t-1]])
        return S



def main():
    ticker = "PLTR"
    df = yf.Ticker(ticker).history(period="5y")
    prices = df["Close"]
    logreturns = np.log(prices / prices.shift(1))

    feat = pd.DataFrame(index=prices.index)
    feat["logret"]   = logreturns
    feat["ret_3m"]   = prices.pct_change(63)   
    feat["ret_6m"]   = prices.pct_change(126) 
    feat["ret_12m"]  = prices.pct_change(252) 



    feat = feat.dropna()


    logreturns = feat["logret"].values       
    extra_feats = feat[["ret_3m","ret_6m","ret_12m"]].values  

    split = int(len(logreturns) * 0.9)
    train_returns = logreturns[:split]
    test_returns  = logreturns[split:]
    train_extra = extra_feats[:split]
    
    n_steps = len(test_returns)

    detector = HmmDetect(train_returns, means=3)
    h = calibrate_heston(prices[:split], r=0.01, q=0.0)

    regimes_train = detector.model.predict(train_returns.reshape(-1, 1))

    ml_drift = MLDrift(
        window=20,
        n_regimes=detector.model.n_components,
    )
    ml_drift.fit(train_returns, regimes_train, train_extra)

    S_reg = detector.sim_reg_path(n_steps)
    drift_path = ml_drift.make_drift_path(
        train_logreturns=train_returns,
        S_reg=S_reg,
        fallback_mu=train_returns.mean()
    )  

    n_paths = 20000
    P0 = prices.iloc[split]         

    mu_hat = drift_path.mean()
    mu_real = train_returns.mean()

    drift_path_corrected = drift_path - mu_hat + mu_real
    drift_path_annual = drift_path_corrected * 252.0

    S_paths, v_paths = h.simulate_paths(
        S0=P0,
        T=1.0,                
        steps=n_steps,
        n_paths=n_paths,
        q=0.0,
        mu_override=drift_path_annual
    )


    sim_prices = S_paths[:, 1:]      


    median_curve = np.median(sim_prices, axis=0) * 1

    dists = np.sum((sim_prices - median_curve)**2, axis=1)
    idx = np.argmin(dists)
    central_path = sim_prices[idx]   


    central_returns = np.diff(central_path) / central_path[:-1]
    mean_daily = central_returns.mean()
    std_daily  = central_returns.std()
    sharpe = (mean_daily / std_daily) * np.sqrt(252)
    print("Estimated Sharpe:", sharpe)

    actual_prices = prices.iloc[split+1 : split+1+n_steps]


    central_path_plot = np.insert(central_path, 0, P0)

    plt.figure(figsize=(12,6))
    plt.plot(actual_prices.index.insert(0, prices.index[split]), central_path_plot,
             color='blue', label=f"Medoid Path")
    plt.plot(actual_prices.index.insert(0, prices.index[split]),
             np.insert(actual_prices.values, 0, P0),
             label="Actual", color="black")
    plt.legend()
    plt.grid(True)
    plt.show()




if __name__ == "__main__":
    main()


