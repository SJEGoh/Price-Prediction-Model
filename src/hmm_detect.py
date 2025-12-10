from hmmlearn.hmm import GaussianHMM
from options import calibrate_heston
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from ML import MLDrift
import pandas as pd
from helper import earnings
import shap

class HmmDetect:
    def __init__(self, logreturns_data, means = 3):
        self.model = GaussianHMM(n_components=means, covariance_type="full", n_iter=1000)
        self.model.fit(logreturns_data.reshape(-1, 1))

        _, posteriors = self.model.score_samples(logreturns_data.reshape(-1, 1))

        self.mu = self.model.means_.flatten()
        self.sigma = np.sqrt(self.model.covars_).flatten()
        self.P = self.model.transmat_
        self.pi_last = posteriors[-1] 

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
    ticker = "Z74.SI"
    t = yf.Ticker(ticker)
    df = t.history(period="5y", auto_adjust=True)
    spy = yf.Ticker("STI").history(period="5y")["Close"]
    prices = df["Close"]
    split = int(len(prices) * 0.95)
    train_price = prices[:split]
    train_spy = spy[:split]

    logreturns = np.log(prices / prices.shift(1))
    train_log = np.log(train_price/train_price.shift(1))

    feat = pd.DataFrame(index=train_price.index)
    feat["logret"]   = train_log
    feat["ret_3m"]   = train_price.pct_change(21)   
    feat["ret_6m"]   = train_price.pct_change(63) 
    feat["ret_12m"]  = train_price.pct_change(126) 
    feat["month"] = feat.index.month
    feat["dow"]   = feat.index.dayofweek
    feat["volume"] = df[:split]["Volume"]
    feat["vol_20d_avg"] = feat["volume"].rolling(20).mean()
    feat["vol_z_20d"] = (feat["volume"] - feat["vol_20d_avg"]) / feat["volume"].rolling(20).std()
    feat["range"] = (df[:split]["High"] - df[:split]["Low"])/df[:split]["Open"]
    feat["rv_21"] = train_log.rolling(10).std()
    feat["rv_63"] = train_log.rolling(40).std()
    feat["rv_252"] = train_log.rolling(80).std()
    feat["vol_ratio_21_63"]  = feat["rv_21"]  / feat["rv_63"]
    feat["vol_ratio_21_252"] = feat["rv_21"]  / feat["rv_252"]
    feat["dividends"] = df[:split]["Dividends"]
    feat["skew_21"] = train_log.rolling(21).skew()
    feat["kurt_21"] = train_log.rolling(21).kurt()
    feat["mkt_ret"] = np.log(train_spy / train_spy.shift(1))
    feat["mkt_vol"] = feat["mkt_ret"].rolling(21).std()
    feat["stock_split"] = df[:split]["Stock Splits"]
    feat["ma_20"] = train_price.rolling(20).mean()
    feat["ma_20_dist"] = train_price / feat["ma_20"] - 1
    feat["ma_50"] = train_price.rolling(50).mean()
    feat["ma_50_dist"] = train_price / feat["ma_50"] - 1
    feat["ma_200"] = train_price.rolling(200).mean()
    feat["ma_200_dist"] = train_price / feat["ma_200"] - 1
    ma = train_price.rolling(20).mean()
    std = train_price.rolling(20).std()
    feat["pct_boll"] = (train_price - (ma - 2*std)) / (4*std)

    
    feat = pd.get_dummies(
        feat,
        columns=["month", "dow"],
        prefix=["month", "dow"],
        drop_first=False
    )
    feature_cols = [c for c in feat.columns if c != "logret"]
    try:
        earn = t.get_earnings_dates().reset_index()
        feat = earnings(earn, feat)
        feature_cols = feature_cols + ["eps_actual", "eps_est", "eps_surprise", "eps_negative", "eps_positive"]
    except:
        print("No Earnings Data")

    feat = feat.fillna(0)
    train_returns = feat["logret"].shift(-1).dropna().values
    train_extra = feat[feature_cols].iloc[:-1].fillna(0).values

    test_returns = logreturns[split:].iloc[1:].values
    n_steps = len(test_returns)

    detector = HmmDetect(train_returns, means=3)
    
    h = calibrate_heston(prices[:split], r=0.01, q=0.0)
    
    regimes_train = detector.model.predict(train_returns.reshape(-1, 1))

    ml_drift = MLDrift(
        window=40,
        n_regimes=detector.model.n_components,
    )
    ml_drift.fit(train_returns, regimes_train, train_extra)
    explainer = shap.TreeExplainer(ml_drift.model)
    shap_values = explainer.shap_values(train_extra, check_additivity=False)

    shap.summary_plot(shap_values, train_extra, feature_names=feature_cols)
    
    S_reg = detector.sim_reg_path(n_steps)
    drift_path = ml_drift.make_drift_path(
        train_logreturns=train_returns,
        S_reg=S_reg,
        fallback_mu=train_returns.mean(),
        extra_feats_train = train_extra
    )  

    n_paths = 40000
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


