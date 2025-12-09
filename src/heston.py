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
    def _simpson_weights(self, N: int):
        if N % 2 != 0:
            raise ValueError("N must be even for Simpson weights.")
        w = np.ones(N)
        w[1:N-1:2] = 4
        w[2:N-2:2] = 2
        return w
    def heston_fft_calls(
            self,
            S0: float,
            T: float,
            q: float,
            N: int = 4096,
            eta: float = 0.25,
            alpha: float = 1.50
    ):
        n = np.arange(N)
        v = eta * n

        i = 1j
        phi_shift = self.heston_cf(v - (i * (alpha+1)) , T, S0, q)
        denom = (alpha**2 + alpha - v**2 + i*(2*alpha + 1)*v)
        psi = np.exp(-self.r * T) * phi_shift / denom

        w = self._simpson_weights(N) * (eta / 3.0)


        lam = 2.0 * np.pi / (N * eta)   
        b   = 0.5 * N * lam          
        x   = psi * np.exp(1j * b * v) * w

        F   = np.fft.fft(x)
        F   = np.real(F)
        j = np.arange(N)
        k = -b + j * lam          
        K = np.exp(k)

        calls = np.exp(-alpha * k) / np.pi * F
        order = np.argsort(K)
        return K[order], np.maximum(calls[order], 0.0)
    
    def heston_price_at_strikes(self,
        strikes, S0, T,
        q, alpha=1.5, N=4096, eta=0.25
    ):
        K_grid, C_grid = self.heston_fft_calls(
            S0, T, q,
            alpha=alpha, N=N, eta=eta
        )
        prices = np.interp(strikes, K_grid, C_grid)

        return prices
    def heston_fft_call_price(self,
        S0: float, K: float, T: float,  q: float,
        N: int = 4096, eta: float = 0.25, alpha: float = 1.5
    ):

        K_grid, C_grid = self.heston_fft_calls(S0, T, q, N=N, eta=eta, alpha=alpha)
        if K <= K_grid[0]:
            return C_grid[0]
        if K >= K_grid[-1]:
            return C_grid[-1]
        idx = np.searchsorted(K_grid, K)
        x0, x1 = K_grid[idx-1], K_grid[idx]
        y0, y1 = C_grid[idx-1], C_grid[idx]
        return y0 + (y1 - y0) * (K - x0) / (x1 - x0)

def main():
    S0, r, q, T = 100.0, 0.01, 0.00, 1.0
    h = Heston(kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04, r = r)
    K, C = h.heston_fft_calls(S0, T, q, N=4096, eta=0.25, alpha=1.5)
    call_100 = h.heston_fft_call_price(S0, 100, T, q)

    print(call_100)

if __name__ == "__main__":
    main()

