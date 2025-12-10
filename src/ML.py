import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

class MLDrift:
    def __init__(self, window=20, n_regimes=3):
        self.window = window
        self.n_regimes = n_regimes
        self.model = HistGradientBoostingRegressor(
            max_depth=5,
            learning_rate=0.05,
            max_iter=500,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            random_state=42,
        )
        self._fitted = False

    def _regime_onehot(self, r_label):
        oh = np.zeros(self.n_regimes, dtype=float)
        if 0 <= r_label < self.n_regimes:
            oh[int(r_label)] = 1.0
        return oh

    def fit(self, logreturns, regimes, extra_feats=None):
        r   = np.asarray(logreturns).reshape(-1)
        reg = np.asarray(regimes).reshape(-1)

        if extra_feats is not None:
            extra_feats = np.asarray(extra_feats)
            assert len(extra_feats) == len(r)

        w = self.window
        X, y = [], []

        for t in range(w, len(r)):
            feat_returns = r[t-w:t]              
            feat_regime  = self._regime_onehot(reg[t])

            if extra_feats is not None:
                feat_extra = extra_feats[t]       
                feats = np.concatenate([feat_returns, feat_regime, feat_extra])
            else:
                feats = np.concatenate([feat_returns, feat_regime])

            X.append(feats)
            y.append(r[t])

        X = np.stack(X)
        y = np.array(y)
        sample_weight = np.exp(np.linspace(-3, 0, len(y)))
        self.model.fit(X, y, sample_weight = sample_weight)
        self._fitted = True

    def make_drift_path(self, train_logreturns, S_reg, extra_feats_train=None, fallback_mu=None):
        assert self._fitted
        r_hist = np.asarray(train_logreturns).reshape(-1)
        w = self.window
        n_steps = len(S_reg)

        if fallback_mu is None:
            fallback_mu = r_hist.mean()

        window_vec = r_hist[-w:].copy()

        if extra_feats_train is not None:
            extra_feats_train = np.asarray(extra_feats_train)
            last_extra = extra_feats_train[-1]
        else:
            last_extra = None

        drift_path = np.zeros(n_steps)

        for t in range(n_steps):
            reg_label = int(S_reg[t])
            feat_returns = window_vec
            feat_regime  = self._regime_onehot(reg_label)

            if last_extra is not None:
                feat_extra = last_extra
                X_t = np.concatenate([feat_returns, feat_regime, feat_extra]).reshape(1, -1)
            else:
                X_t = np.concatenate([feat_returns, feat_regime]).reshape(1, -1)

            try:
                pred = float(self.model.predict(X_t)[0])
            except Exception:
                pred = float(fallback_mu)

            drift_path[t] = pred

            window_vec = np.roll(window_vec, -1)
            window_vec[-1] = pred

        return drift_path
