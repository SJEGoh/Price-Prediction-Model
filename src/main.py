import matplotlib.pyplot as plt
from helper import make_features, load_data, filter_feat, get_drift_corrected, get_central_path
from hmm_detect import HmmDetect
from options import calibrate_heston
from ML import MLDrift
from benchmark import GBM, Garch
import numpy as np


def main():
    ticker = "NVDA"
    split = 0.9
    train_price, test_price, log_train_ser, log_test_ser, df_train, t = load_data(ticker, period="10y", split=split)
    rfr = 0.02
    n_paths = 40000
    log_train = log_train_ser.dropna().values
    log_test = log_test_ser.dropna().values

    n_steps = len(log_test)
    q, feat = make_features(df_train, train_price, log_train_ser, t)

    feature_cols = [c for c in feat.columns if c != "logret"]
    train_extra = feat[feature_cols].iloc[:-1].fillna(0).values

    HMMdetector = HmmDetect(log_train, means = 3)

    h = calibrate_heston(train_price, r = rfr, q = q)


    regimes_train = HMMdetector.model.predict(log_train.reshape(-1, 1))

    ml_drift = MLDrift(
        window = 20,
        n_regimes = HMMdetector.model.n_components
    )

    ml_drift.fit(log_train, regimes_train, train_extra)

    filtered_feat = filter_feat(ml_drift.model,feature_cols, train_extra, feat, threshold = 0.005)
    S_reg = HMMdetector.sim_reg_path(n_steps)
    P0 = test_price.iloc[0]

    fft_call, mc_call, _, _ = h.heston_fft_terminal_median(
    S0=P0,
    T=n_steps/252,
    q=q,
    )
    print("FFT call:", fft_call)

    drift_path_corrected = get_drift_corrected(ml_drift, log_train, regimes_train, filtered_feat, S_reg)

    _, v_tmp = h.simulate_paths(
        S0=P0,
        T=1.0,                
        steps=n_steps,
        n_paths=n_paths,
        q=q,
        mu_override=None
    )

    actual_v = train_price.std()
    vol_scale = actual_v/np.mean(np.sqrt(v_tmp))
    print(f"Volatility scale: {vol_scale}")

    S_paths, _ = h.simulate_paths(
        S0=P0,
        T=1.0,                
        steps=n_steps,
        n_paths=n_paths,
        q=q,
        mu_override=drift_path_corrected,
        vol_scale = vol_scale
    )

    central_path, lower, upper = get_central_path(S_paths, P0, terminal = fft_call)

    GBM_model = GBM(log_train.mean(), log_train.std())
    gbm_paths = GBM_model.simulate(S0 = P0, n_steps = n_steps, n_paths = n_paths)
    gbm_central_path, gbm_lower, gbm_upper = get_central_path(gbm_paths, P0)

    GARCH_model = Garch()
    GARCH_model.fit(log_train)
    garch_paths = GARCH_model.simulate(S0 = P0, n_steps = n_steps, n_paths = n_paths)
    garch_central_path, garch_lower, garch_upper = get_central_path(garch_paths, P0)


    plt.figure(figsize = (12,6))
    plt.plot(test_price.index, test_price.values, color="black", label="Actual")
    plt.plot(test_price.index, central_path,
            color='blue', label=f"Medoid Path")
    plt.plot(test_price.index, gbm_central_path,
            color='red', label=f"GBM Medoid Path", alpha = 0.5, linestyle = ":")
    plt.plot(test_price.index, garch_central_path,
        color='green', label=f"Garch Medoid Path", alpha = 0.5, linestyle = "--")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
