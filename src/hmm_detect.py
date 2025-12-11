from hmmlearn.hmm import GaussianHMM
from options import calibrate_heston
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from ML import MLDrift
import pandas as pd
from helper import earnings_data
import shap
from arch import arch_model

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

def simulate_gbm_paths(S0, mu, sigma, n_steps, n_paths, dt=1/252):
    """
    Simple GBM Monte Carlo:
        dS/S = mu dt + sigma dW
    Returns array of shape (n_paths, n_steps+1) including S0 at t=0.
    """
    # Normal shocks: (n_paths, n_steps)
    Z = np.random.randn(n_paths, n_steps)
    # increments in log-price
    drift = (mu - 0.5 * sigma**2) * dt
    diff  = sigma * np.sqrt(dt) * Z

    log_S = np.zeros((n_paths, n_steps+1))
    log_S[:, 0] = np.log(S0)

    for t in range(n_steps):
        log_S[:, t+1] = log_S[:, t] + drift + diff[:, t]

    return np.exp(log_S)

def simulate_garch_paths(S0, sigma_t, n_paths):
    """
    Simulate price paths under a GARCH vol forecast with zero drift:
        r_t ~ N(0, sigma_t^2)
        S_t = S_{t-1} * exp(r_t)
    sigma_t: (n_steps,) array of daily forecasted vol
    Returns: (n_paths, n_steps+1) prices including S0 at t=0.
    """
    sigma_t = np.asarray(sigma_t)
    n_steps = len(sigma_t)

    Z = np.random.randn(n_paths, n_steps)   # shocks
    R = sigma_t * Z                         # returns

    log_S = np.zeros((n_paths, n_steps+1))
    log_S[:, 0] = np.log(S0)

    for t in range(n_steps):
        log_S[:, t+1] = log_S[:, t] + R[:, t]

    return np.exp(log_S)

def main():
  ...



if __name__ == "__main__":
    main()


