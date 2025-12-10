import numpy as np
import yfinance as yf

class Heston:
    def __init__(self, kappa, theta, sigma, rho, v0, r = 0.0):
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0
        self.r = r
    
    def heston_cf(self,u, T, S0, q):
        i = 1j
        x0 = np.log(S0)
        a = self.kappa * self.theta
        b = self.kappa - self.rho * self.sigma * i * u
        d = np.sqrt(b*b +(self.sigma**2) * (i*u + u*u))
        g = (b-d)/(b+d)

        eDT = np.exp(-d * T)

        one_minus_g_eDT = 1 - g * eDT
        one_minus_g     = 1 - g

        one_minus_g_eDT = np.where(np.abs(one_minus_g_eDT) < 1e-15, 1e-15, one_minus_g_eDT)
        one_minus_g     = np.where(np.abs(one_minus_g)     < 1e-15, 1e-15, one_minus_g)

        phi1 = i*u*(self.r - q)*T + (a/(self.sigma**2)) * ((b - d)*T - 2.0*np.log(one_minus_g_eDT/one_minus_g))
        phi2 = ((b - d)/(self.sigma**2)) * ((1 - eDT) / one_minus_g_eDT)

        return np.exp(phi1 + phi2 * self.v0 + i * u * x0)

    def simulate_paths(self, S0, T, steps, n_paths, q = 0.0, mu_override = None):
        dt = T/steps
        S = np.zeros((n_paths, steps+1))
        v = np.zeros((n_paths, steps+1))

        S[:, 0] = S0
        v[:, 0] = self.v0

        for t in range(steps):
            z1 = np.random.normal(size=n_paths)
            z2 = np.random.normal(size=n_paths)
            z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2
            v_prev = v[:,t]
            v_new = np.maximum(
                v_prev + self.kappa*(self.theta - v_prev)*dt +
                self.sigma*np.sqrt(np.maximum(v_prev, 0))*np.sqrt(dt)*z2, 0
                )
            v[:,t+1] = v_new
            
            if mu_override is None:
                drift = (self.r - q)
            else:
                drift = mu_override[t]

            S[:,t+1] = S[:,t] * np.exp(
                (drift - 0.5*v_prev)*dt +
                np.sqrt(v_prev*dt)*z1
                )
        return S, v
    
def main():
    S0, r, q, T = 100.0, 0.01, 0.00, 1.0
    h = Heston(kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04, r = r)
    K, C = h.heston_fft_calls(S0, T, q, N=4096, eta=0.25, alpha=1.5)
    call_100 = h.heston_fft_call_price(S0, 100, T, q)

    print(call_100)

if __name__ == "__main__":
    main()

