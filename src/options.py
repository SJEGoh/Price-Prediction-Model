import numpy as np
from scipy.optimize import minimize
from heston import Heston   # your existing Heston class


def _realized_variance_from_returns(logreturns, window=20):
    """Rolling realized variance (annualized) from daily log returns."""
    # daily squared returns
    sq = logreturns**2
    # rolling mean of squared returns
    rv = np.convolve(sq, np.ones(window)/window, mode="valid")
    # annualize
    rv_annual = rv * 252.0
    return rv_annual  # length = len(logreturns) - window + 1


def _cir_neg_loglik(params, v, dt):
    """
    Quasi negative log-likelihood for CIR using Euler approximation:
    v_{t+1} ~ N(m_t, sigma^2 * v_t * dt)
    """
    kappa, theta, sigma = params

    # basic sanity: enforce positivity
    if kappa <= 0 or theta <= 0 or sigma <= 0:
        return 1e9

    v_t = v[:-1]
    v_tp1 = v[1:]

    m_t = v_t + kappa * (theta - v_t) * dt
    var_t = sigma**2 * v_t * dt

    # avoid division by 0
    var_t = np.where(var_t < 1e-12, 1e-12, var_t)

    resid = v_tp1 - m_t
    ll = -0.5 * (np.log(2 * np.pi * var_t) + (resid**2) / var_t)
    return -np.sum(ll)


def calibrate_heston(prices, r=0.01, q=0.0,
                                 rv_window=10, dt=1/252):
    logret = np.log(prices / prices.shift(1)).dropna().values

    v_series = _realized_variance_from_returns(logret, window=rv_window)

    logret_cut = logret[rv_window-1:] 

    v_mean = v_series.mean()
    v_var = v_series.var()

    kappa0 = 2.0
    theta0 = max(v_mean, 1e-6)
    sigma0 = np.sqrt(2 * kappa0 * theta0)  

    x0 = np.array([kappa0, theta0, sigma0])


    obj = lambda p: _cir_neg_loglik(p, v_series, dt)

    res = minimize(
        obj,
        x0,
        method="L-BFGS-B",
        options={"maxiter": 300}
    )

    if not res.success:
        print(res.message)

    kappa, theta, sigma = res.x

    dv = np.diff(v_series)
    lr_for_rho = logret_cut[1:len(dv)+1]
    if len(dv) > 10:
        rho_est = np.corrcoef(lr_for_rho, dv)[0, 1]
    else:
        rho_est = -0.5  

    rho = float(np.clip(rho_est, -0.95, 0.95))


    v0 = float(v_series[-1])

    print("Heston Params: ")
    print(f"kappa = {kappa}")
    print(f"theta = {theta}")
    print(f"sigma = {sigma}")
    print(f"rho   = {rho}")
    print(f"v0    = {v0}")

    h = Heston(
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        rho=rho,
        v0=v0,
        r=r
    )

    return h
