import pandas as pd
import numpy as np
import yfinance as yf
import shap

def load_data(ticker, period, split):
    t = yf.Ticker(ticker)
    df = t.history(period= period, auto_adjust=True)
    prices = df["Close"]
    split = int(len(prices) * split)

    train_price = prices[:split]
    test_price = prices[split:]

    log_train_ser = np.log(train_price/train_price.shift(1))
    log_test_ser = np.log(test_price/test_price.shift(1))
    return train_price, test_price, log_train_ser, log_test_ser, df[:split], t

def earnings_data(earn_df, feat):
    feat.index = feat.index.normalize()
    earn_df["Earnings Date"] = pd.to_datetime(earn_df["Earnings Date"]).dt.normalize()
    earn_df = earn_df.sort_values("Earnings Date")
    earn_daily = pd.DataFrame(index=feat.index)
    earn_daily = earn_daily.join(earn_df.set_index("Earnings Date"), how="left")

    earn_daily["Reported EPS"] = earn_daily["Reported EPS"].ffill()
    earn_daily["EPS Estimate"] = earn_daily["EPS Estimate"].ffill()
    earn_daily["Surprise(%)"] = earn_daily["Surprise(%)"].ffill()

    feat["eps_actual"] = earn_daily["Reported EPS"]
    feat["eps_est"] = earn_daily["EPS Estimate"]
    feat["eps_surprise"] = earn_daily["Surprise(%)"]

    feat["eps_positive"] = (feat["eps_surprise"] > 0).astype(int)
    feat["eps_negative"] = (feat["eps_surprise"] < 0).astype(int)

    feat.fillna({"eps_actual":-999}, inplace=True)
    feat.fillna({"eps_est":-999}, inplace=True)
    feat.fillna({"eps_surprise":-999}, inplace=True)

    feat.fillna({"eps_positive":0}, inplace=True)
    feat.fillna({"eps_negative":0}, inplace=True)

    return feat


def make_features(df, train_price, train_log, t):
    l = len(train_price)
    if l > 3000:
        days_return = 252
        days_rv = 252
    elif l > 1500:
        days_return = 176
        days_rv = 176
    else:
        days_return = 60
        days_rv = 60 

    feat = pd.DataFrame(index=train_price.index)
    feat["logret"] = train_log
    feat["1_trail_return"] = feat["logret"].shift(1)
    feat["trail_return"] = feat["logret"].rolling(20).mean().shift(1)
    feat["40_trail_return"] = feat["logret"].rolling(60).mean().shift(1)
    feat["_trail_return"] = feat["logret"].rolling(180).mean().shift(1)
    feat["ret_3m"]   = train_price.pct_change(days_return//4)   
    feat["ret_6m"]   = train_price.pct_change(days_return//2) 
    feat["ret_12m"]  = train_price.pct_change(days_return) 
    feat["month"] = feat.index.month
    feat["dow"]   = feat.index.dayofweek
    feat["volume"] = df["Volume"]
    feat["vol_20d_avg"] = feat["volume"].rolling(20).mean()
    feat["vol_z_20d"] = (feat["volume"] - feat["vol_20d_avg"]) / feat["volume"].rolling(20).std()
    feat["garman_klass"] = (
        0.5 * (np.log(df["High"] / df["Low"]) ** 2)
        - (2*np.log(2) - 1) * (np.log(df["Close"] / df["Open"]) ** 2)
    )
    feat["rv_21"] = train_log.rolling(days_rv//12).std()
    feat["rv_63"] = train_log.rolling(days_rv//4).std()
    feat["rv_252"] = train_log.rolling(days_rv).std()
    feat["vol_ratio_21_63"]  = feat["rv_21"]  / feat["rv_63"]
    feat["vol_ratio_21_252"] = feat["rv_21"]  / feat["rv_252"]
    feat["dividends"] = df["Dividends"]
    q = sum(feat["dividends"])/len(feat) 
    feat["skew_21"] = train_log.rolling(21).skew()
    feat["kurt_21"] = train_log.rolling(21).kurt()

    feat["stock_split"] = df["Stock Splits"]
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

    try:
        feat = earnings_data(t.get_earnings_dates()[len(df):].reset_index(), feat)
    except:
        print("No Earnings Data")

    feat = feat.fillna(0)

    return q, feat

def filter_feat(model, feature_cols,train_extra, feat, K = None, threshold = None):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(train_extra, check_additivity=False)
    shap_abs_mean = np.mean(np.abs(shap_values), axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": shap_abs_mean
    }).sort_values("importance", ascending=False)
    if K:
        filtered_extra = importance_df[importance_df["importance"] > importance_df.iloc[K]["importance"]]["feature"]
    if threshold:
        filtered_extra = importance_df[importance_df["importance"] > threshold]["feature"]
        
    filtered_feat = feat[filtered_extra.values].iloc[:-1].values

    return filtered_feat

def get_drift_corrected(model, log_train, regimes_train, filtered_feat, S_reg):
    model.fit(log_train, regimes_train, filtered_feat)
    
    simmed_drift_path = model.make_drift_path(
        train_logreturns=log_train,
        S_reg=S_reg,
        fallback_mu=log_train.mean(),
        extra_feats_train = filtered_feat
    )  

    drift_train = model.make_drift_path(
        train_logreturns=log_train,
        S_reg=regimes_train,       
        fallback_mu=log_train.mean(),
        extra_feats_train=filtered_feat
    )
    shrink_factor = log_train.std()/drift_train.std()
    print("Shrink factor:", shrink_factor)

    mu_hat = drift_train.mean()
    mu_real = log_train.mean()
    
    drift_path_corrected = shrink_factor * (simmed_drift_path - mu_hat + mu_real)

    return drift_path_corrected


def get_central_path(S_paths, P0, terminal=None, z_band=1.0):
    S_paths = np.asarray(S_paths)
    if S_paths.ndim != 2:
        raise ValueError(f"S_paths must be 2D (n_paths, n_steps+1), got {S_paths.shape}")

    sim_prices = S_paths[:, 1:]  

    if terminal is not None:
        terminal_prices = sim_prices[:, -1]
        sigma = terminal_prices.std()

        if sigma == 0:
            mask = np.ones_like(terminal_prices, dtype=bool)
        else:
            z = (terminal_prices - terminal) / sigma
            mask = np.abs(z) <= z_band

        if not mask.any():
            candidates = sim_prices
            idx_candidates = np.arange(sim_prices.shape[0])
        else:
            candidates = sim_prices[mask]
            idx_candidates = np.where(mask)[0]

        if candidates.ndim == 1:
            candidates = candidates[np.newaxis, :]

        median_curve = np.median(candidates, axis=0)
        dists_local = np.sum((candidates - median_curve) ** 2, axis=1)
        idx_local = np.argmin(dists_local)
        idx = idx_candidates[idx_local]

        central_path = sim_prices[idx]
        central_path = np.insert(central_path, 0, P0)
        return central_path, median_curve, idx
    median_curve = np.median(sim_prices, axis=0)
    dists = np.sum((sim_prices - median_curve)**2, axis=1)
    idx = np.argmin(dists)
    central_path = sim_prices[idx]
    central_path = np.insert(central_path, 0, P0)

    lower = np.percentile(sim_prices, 40, axis=0)   # (n_steps,)
    upper = np.percentile(sim_prices, 60, axis=0) 

    return central_path, lower, upper

