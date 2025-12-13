import numpy as np

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
        one_minus_g = 1 - g

        one_minus_g_eDT = np.where(np.abs(one_minus_g_eDT) < 1e-15, 1e-15, one_minus_g_eDT)
        one_minus_g = np.where(np.abs(one_minus_g) < 1e-15, 1e-15, one_minus_g)

        phi1 = i*u*(self.r - q)*T + (a/(self.sigma**2)) * ((b - d)*T - 2.0*np.log(one_minus_g_eDT/one_minus_g))
        phi2 = ((b - d)/(self.sigma**2)) * ((1 - eDT) / one_minus_g_eDT)

        return np.exp(phi1 + phi2 * self.v0 + i * u * x0)

    def simulate_paths(self, S0, T, steps, n_paths, q = 0.0, mu_override = None, vol_scale = 1.0):
        dt = T/steps
        log_S = np.zeros((n_paths, steps+1))
        S = np.zeros((n_paths, steps+1))
        v = np.zeros((n_paths, steps+1))
        log_S[:, 0] = np.log(S0)
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

            vol_term = vol_scale * np.sqrt(np.maximum(v_prev, 0.0) * dt)
            drift_term = (drift - 0.5 * (vol_scale**2) * v_prev) * dt
            log_S[:, t+1] = log_S[:, t] + drift_term + vol_term * z1
            
        return log_S, v
    
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
        alpha: float = 1.5  
    ):
        n = np.arange(N)
        v = eta * n 

        i = 1j

        phi_shift = self.heston_cf(v - (alpha + 1)*i, T, S0, q)
        denom = (alpha**2 + alpha - v**2 + i*(2*alpha + 1)*v) 
        psi = np.exp(-self.r*T) * phi_shift / denom


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
    def heston_fft_call_price(
        self,
        S0: float,
        K0: float,
        T: float,
        q: float,
        N: int = 4096,
        eta: float = 0.25,
        alpha: float = 1.5
    ):
        """
        Price a single call C(S0, K0, T) via Carr–Madan FFT.
        """
        K, C = self.heston_fft_calls(
            S0=S0,
            T=T,
            q=q,
            N=N,
            eta=eta,
            alpha=alpha
        )
        return np.interp(K0, K, C)
    
    
    def heston_fft_terminal_median(self, S0, T, q, N=4096, eta=0.25, alpha=1.5):
        r = self.r
        K, C = self.heston_fft_calls(S0=S0, T=T, q=q, N=N, eta=eta, alpha=alpha)

        dC_dK  = np.gradient(C, K)
        d2C_dK = np.gradient(dC_dK, K)
        pdf_raw = np.exp(r * T) * d2C_dK
        pdf = np.maximum(pdf_raw, 0.0)

        mask = (K > 0.5 * S0) & (K < 2.0 * S0)
        K_m = K[mask]
        pdf_m = pdf[mask]

        area = np.trapz(pdf_m, K_m)
        if area > 0:
            pdf_m /= area

        cdf = np.cumsum(pdf_m) * (K_m[1] - K_m[0])
        median_idx = np.searchsorted(cdf, 0.5)
        median_idx = np.clip(median_idx, 0, len(K_m) - 1)
        K_median = K_m[median_idx]
        return K_median, K_m, pdf_m, cdf


def main():
    S0, r, q, T = 100.0, 0.01, 0.00, 1.0
    h = Heston(kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04, r = r)
    K, C = h.heston_fft_calls(S0, T, q, N=4096, eta=0.25, alpha=1.5)
    call_100 = h.heston_fft_call_price(S0, 100, T, q)

    print(call_100)

if __name__ == "__main__":
    main()

