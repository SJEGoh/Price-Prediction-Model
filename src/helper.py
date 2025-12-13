import pandas as pd
import numpy as np
from hmm_detect import HmmDetect
from ML import MLDrift
from benchmark import GBM, Garch
from options import calibrate_heston
import shap
from scipy.stats import wasserstein_distance

def load_data(df, split):
    prices = df["Close"]
    split = int(len(prices) * split)

    train_price = prices[:split]
    test_price = prices[split:]

    log_train_ser = np.log(train_price/train_price.shift(1))
    log_test_ser = np.log(test_price/test_price.shift(1))
    return train_price, test_price, log_train_ser, log_test_ser, df[:split]

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
    feat["season"] = np.where(feat["month"] <= 3, 0, 
                              np.where((feat["month"] > 3) & (feat["month"] <= 6), 1,
                                        np.where((feat["month"] > 6) & (feat["month"] <= 9), 2, 3)))
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
        columns=["month", "dow", "season"],
        prefix=["month", "dow", "season"],
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

    return filtered_feat, filtered_extra.values

def get_drift_corrected(model, log_train, regimes_train, filtered_feat, S_reg, hmm_means, hmm_sds):
    model.fit(log_train, regimes_train, filtered_feat)
    
    simmed_drift_path = model.make_drift_path(
        train_logreturns=log_train,
        S_reg=regimes_train,
        fallback_mu=log_train.mean(),
        extra_feats_train = filtered_feat
    )  

    residuals = log_train[-len(simmed_drift_path):] - simmed_drift_path
    sigma_ml = np.std(residuals) 

    drift_train = model.make_drift_path(
        train_logreturns=log_train,
        S_reg=S_reg,       
        fallback_mu=log_train.mean(),
        extra_feats_train=filtered_feat
    )

    drift_posterior = np.zeros_like(drift_train)
    for t in range(len(drift_train)):
        reg = S_reg[t]
        mu_k = hmm_means[reg]
        sigma_k = hmm_sds[reg]

        # precision-based weight (1/var)
        prec_ml = 1.0 / (sigma_ml**2 + 1e-12)
        prec_k  = 1.0 / (sigma_k**2 + 1e-12)

        alpha = prec_ml / (prec_ml + prec_k)  # weight on ML

        drift_posterior[t] = alpha * drift_train[t] + (1 - alpha) * mu_k

    return drift_posterior


def get_central_path(S_paths, P0, terminal=None, z_band=1.0):
    # fix later
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
        lower = np.percentile(candidates, 10, axis=0)   
        lower = np.insert(lower, 0, P0)
        upper = np.percentile(candidates, 90, axis=0) 
        upper = np.insert(upper, 0, P0)
        return central_path, lower, upper
    median_curve = np.median(sim_prices, axis=0)
    dists = np.sum((sim_prices - median_curve)**2, axis=1)
    idx = np.argmin(dists)
    central_path = sim_prices[idx]
    central_path = np.insert(central_path, 0, P0)

    lower = np.percentile(sim_prices, 10, axis=0)   
    lower = np.insert(lower, 0, P0)
    upper = np.percentile(sim_prices, 90, axis=0) 
    upper = np.insert(upper, 0, P0)

    return central_path, lower, upper

def get_hist(values, bins=3, value_range=None):
    hist, bin_edges = np.histogram(values, bins=bins, range=value_range)
    hist = hist.astype(float)
    hist /= hist.sum()  
    return hist, bin_edges

def test_metrics(actual_returns, log_S_paths, log_central_path): 
    sim_dist = np.diff(log_S_paths, axis = 1).flatten()
    sim_central = np.diff(log_central_path)[1:]
    a, _ = get_hist(sim_dist)
    a += 1e-12
    b, _ = get_hist(actual_returns)
    b += 1e-12
    c, _ = get_hist(sim_central)
    c += 1e-12
    m = 0.5 * (c + b)
    JS_dist = 0.5 * np.sum(a * np.log(a / m)) + 0.5 * np.sum(b * np.log(b / m))
    JS_med = 0.5 * np.sum(c * np.log(c / m)) + 0.5 * np.sum(b * np.log(b / m))
    w = wasserstein_distance(sim_central, actual_returns)
    RMSE = np.sqrt(np.sum((sim_central - actual_returns)**2)/len(log_central_path))
    return [JS_med, JS_dist, w, RMSE]

def bayesian_confidence(HMMdetector, log_train, drift_path, n_steps):
    K = HMMdetector.P.shape[0]
    p = HMMdetector.model.predict_proba(log_train.reshape(-1, 1))[-1]
    w = np.zeros(n_steps)

    for t in range(n_steps):
        H = -(p * np.log(p + 1e-12)).sum()
        w[t] = 1 - H/np.log(K)
        p = p @ HMMdetector.P
    mu_base = log_train.mean()
    return mu_base + w * (drift_path - mu_base)

def test(train_price, test_price, log_train_ser, log_test_ser, df_train, t):
    rfr = 0.02
    n_paths = 40000
    P0 = test_price.iloc[0]
    log_train = log_train_ser.dropna().values
    log_test = log_test_ser.dropna().values

    n_steps = len(log_test)
    q, feat = make_features(df_train, train_price, log_train_ser, t)

    feature_cols = [c for c in feat.columns if c != "logret"]
    train_extra = feat[feature_cols].iloc[:-1].fillna(0).values

    HMMdetector = HmmDetect(log_train, means = 3)
    hmm_means = HMMdetector.model.means_.flatten()      
    hmm_sds = np.sqrt(HMMdetector.model.covars_.flatten()) 
    regimes_train = HMMdetector.model.predict(log_train.reshape(-1, 1))   
    S_reg = HMMdetector.sim_reg_path(n_steps)

    h, _ = calibrate_heston(train_price, r = rfr, q = q)

    ml_drift = MLDrift(
        window = 20,
        n_regimes = HMMdetector.model.n_components
    )

    ml_drift.fit(log_train, regimes_train, train_extra)

    filtered_feat, _ = filter_feat(ml_drift.model,feature_cols, train_extra, feat, K = 10)

    fft_call, mc_call, _, _ = h.heston_fft_terminal_median(
    S0=P0,
    T=n_steps/252,
    q=q,
    )
    # print("FFT call:", fft_call)

    drift_path_corrected = get_drift_corrected(ml_drift, log_train, regimes_train, filtered_feat, S_reg,
                                               hmm_means, hmm_sds)
    
    mu_eff = bayesian_confidence(HMMdetector, log_train, drift_path_corrected, n_steps)

    _, v_tmp = h.simulate_paths(
        S0=P0,
        T=1.0,                
        steps=n_steps,
        n_paths=n_paths,
        q=q,
        mu_override=None
    )

    log_returns = np.diff(np.log(train_price.values))

    annual_vol = log_returns.std() * np.sqrt(252)

    model_annual_vol = np.mean(np.sqrt(v_tmp)) 

    vol_scale = annual_vol / model_annual_vol
    # print(f"Volatility scale: {vol_scale}")

    log_S_paths, _ = h.simulate_paths(
        S0=P0,
        T=1.0,                
        steps=n_steps,
        n_paths=n_paths,
        q=q,
        mu_override=mu_eff,
        vol_scale = vol_scale
    )

    log_central_path, lower, upper = get_central_path(log_S_paths, np.log(P0), terminal = np.log(fft_call))


    actual_returns = np.diff(log_test)
    model_metrics = test_metrics(actual_returns, log_S_paths, log_central_path)
    central_path = np.exp(log_central_path)

    GBM_model = GBM(log_train.mean(), log_train.std())
    gbm_paths = GBM_model.simulate(S0 = P0, n_steps = n_steps, n_paths = n_paths)
    gbm_central_path, gbm_lower, gbm_upper = get_central_path(gbm_paths, np.log(P0))
    gbm_metrics = test_metrics(actual_returns, gbm_paths, gbm_central_path)
    gbm_central_path = np.exp(gbm_central_path)

    GARCH_model = Garch()
    GARCH_model.fit(log_train)
    garch_paths = GARCH_model.simulate(S0 = P0, n_steps = n_steps, n_paths = n_paths)
    garch_central_path, garch_lower, garch_upper = get_central_path(garch_paths, np.log(P0))
    garch_metrics = test_metrics(actual_returns, garch_paths, garch_central_path)
    garch_central_path = np.exp(garch_central_path)

    return model_metrics, gbm_metrics, garch_metrics, [central_path, gbm_central_path, garch_central_path], test_price

def metrics_summary(df, model_name):
    return pd.DataFrame([{
        "Model": model_name,

        "JS Medoid Mean": f"{df['JS_med'].mean():.3f} ± {df['JS_med'].std():.3f}",
        "JS Medoid Max": df["JS_med"].max(),

        "JS Dist Mean": f"{df['JS_dist'].mean():.3f} ± {df['JS_dist'].std():.3f}",
        "JS Dist Max": df["JS_dist"].max(),

        "Wasserstein Mean": f"{df['Wass'].mean():.3f} ± {df['Wass'].std():.3f}",
        "Wasserstein Max": df["Wass"].max(),

        "RMSE Mean": f"{df['RMSE'].mean():.3f} ± {df['RMSE'].std():.3f}",
        "RMSE Max": df["RMSE"].max(),
    }])

