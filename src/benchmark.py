import numpy as np
from arch import arch_model

class GBM:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def simulate(self, S0, n_steps, n_paths, dt=1.0):
        Z = np.random.randn(n_paths, n_steps)
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        diff  = self.sigma * np.sqrt(dt) * Z

        log_S = np.zeros((n_paths, n_steps+1))
        log_S[:, 0] = np.log(S0)

        for t in range(n_steps):
            log_S[:, t+1] = log_S[:, t] + drift + diff[:, t]

        return log_S


class Garch:
    def __init__(self, p=1, q=1, dist="normal"):
        self.p = p
        self.q = q
        self.dist = dist
        self.model = None
        self.res = None
        self.sigma_forecast = None
        
    def fit(self, train_log):
        garch = arch_model(train_log * 100,
            mean='zero',
            vol='GARCH',
            p=self.p,
            q=self.q,
            dist=self.dist)

        self.res = garch.fit(disp="off")
        self.model = garch

    def simulate(self, S0, n_paths, n_steps):
        garch_forecasts = self.res.forecast(horizon=n_steps, reindex=False)
        var_forecast = garch_forecasts.variance.values[-1]     
        self.sigma_forecast = np.sqrt(var_forecast)   
        Z = np.random.randn(n_paths, n_steps)
        R = self.sigma_forecast * Z / 100

        log_S = np.zeros((n_paths, n_steps+1))
        log_S[:, 0] = np.log(S0)

        for t in range(n_steps):
            log_S[:, t+1] = log_S[:, t] + R[:, t]

        return log_S




