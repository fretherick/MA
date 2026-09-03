import numpy as np
import matplotlib.pyplot as plt

def EulerMilsteinSimWithIntervals(scheme, neg_var, num_sim, n_steps, S0, V0, T, kappa, theta, sigma, r, q, intervals):
    assert scheme in ("Euler", "Milstein")
    assert neg_var in ("Reflect", "Trunca")
    dt = T / n_steps
    S = np.zeros((n_steps + 1, num_sim))
    S[0, :] = S0
    V = np.zeros((n_steps + 1, num_sim))
    V[0, :] = V0
    Vcount0 = 0
    current_t = 0

    for rho, interval_steps in intervals:
        for i in range(num_sim):
            for t in range(current_t + 1, current_t + interval_steps + 1):
                Zv = np.random.randn(1)
                Zs = rho * Zv + np.sqrt(1 - rho**2) * np.random.randn(1)
                if scheme == 'Euler':
                    V[t, i] = V[t-1, i] + kappa * (theta - V[t-1, i]) * dt + sigma * np.sqrt(V[t-1, i]) * np.sqrt(dt) * Zv
                elif scheme == 'Milstein':
                    V[t, i] = V[t-1, i] + kappa * (theta - V[t-1, i]) * dt + sigma * np.sqrt(V[t-1, i]) * np.sqrt(dt) * Zv + 0.25 * sigma**2 * dt * (Zv**2 - 1)
                if V[t, i] <= 0:
                    Vcount0 = Vcount0 + 1
                    if neg_var == 'Reflect':
                        V[t, i] = abs(V[t, i])
                    elif neg_var == 'Trunca':
                        V[t, i] = max(V[t, i], 0)
                S[t, i] = S[t-1, i] * np.exp((r - q - V[t-1, i] / 2) * dt + np.sqrt(V[t-1, i]) * np.sqrt(dt) * Zs)
        current_t += interval_steps

    return S, V, Vcount0

if __name__ == '__main__':
    scheme = "Euler"
    neg_var = "Reflect"
    num_sim = 10
    n_steps = 1000

    S0 = 100
    V0 = 0.09
    T = 1
    kappa = 5.0
    theta = 0.08
    sigma = 0.3
    r = 0.05
    q = 0

    # Define intervals with different rho values (rho, interval_steps)
    intervals = [
        (-.8, 200),   # First interval: negative correlation
        (.8, 200),    # Second interval: positive correlation
        (-.8, 200),   # Third interval: negative correlation
        (-.8, 200),    # Fourth interval: positive correlation
        (-.8, 200)    # Fifth interval: negative correlation
    ]

    # Run simulation
    for j in range(1):
        S, V, _ = EulerMilsteinSimWithIntervals(scheme, neg_var, num_sim, n_steps, S0, V0, T, kappa, theta, sigma, r, q, intervals)

        # Display some simulated paths
        fig, axs = plt.subplots(2, figsize=(10, 6))
        fig.suptitle(f'Heston Model Simulation {j+1}')
        axs[0].plot(S[:, :10])
        axs[0].set_title('Asset Prices')
        axs[0].set_xlabel('Time Steps')
        axs[0].set_ylabel('Price')
        axs[1].plot(V[:, :10])
        axs[1].set_title('Variance')
        axs[1].set_xlabel('Time Steps')
        axs[1].set_ylabel('Variance')
        plt.tight_layout()
        plt.show()
