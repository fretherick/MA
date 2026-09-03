import numpy as np
import numba as nb
import matplotlib.pyplot as plt


# @nb.jit
def EulerMilsteinSim(scheme, neg_var, num_sim, n_steps, rho, S0, V0, T, kappa, theta, sigma, r, q):
    assert scheme in ("Euler", "Milstein")
    assert neg_var in ("Reflect", "Trunca")
    # time step size
    dt = T/n_steps
    # save the sim
    S = np.zeros((n_steps+1, num_sim))
    S[0, :] = S0
    V = np.zeros((n_steps+1, num_sim))
    V[0, :] = V0
    Vcount0 = 0
    for i in range(num_sim):
        for t in range(1, n_steps + 1):
            Zv = np.random.randn(1)
            Zs = rho*Zv + np.sqrt(1 - rho**2)*np.random.randn(1)
            # variance update scheme
            if scheme == 'Euler':
                V[t, i] = V[t-1, i] + kappa * (theta - V[t-1, i]) * dt + sigma*np.sqrt(
                    V[t-1, i])*np.sqrt(dt)*Zv
            elif scheme == 'Milstein':
                V[t, i] = V[t-1, i] + kappa * (theta - V[t-1, i]) * dt + sigma*np.sqrt(
                    V[t-1, i])*np.sqrt(dt)*Zv + 0.25*sigma**2*dt*(Zv**2 - 1)
            # handleing negative variance
            if V[t, i] <= 0:
                Vcount0 = Vcount0 + 1
                if neg_var == 'Reflect':
                    V[t, i] = abs(V[t, i])
                elif neg_var == 'Trunca':
                    V[t, i] = max(V[t, i], 0)

            S[t, i] = S[t-1, i] * np.exp((r - q - V[t-1, i] / 2) * dt + np.sqrt(V[t-1, i]) * np.sqrt(dt) * Zs)
    return S, V, Vcount0


if __name__ == '__main__':
    scheme = "Euler"
    neg_var = "Reflect"
    num_sim = int(1e2)
    n_steps = 100

    rho = -0.5  # correlation
    S0 = 100  # initial stock price
    V0 = 0.09  # initial instantaneous variance
    T = 1
    kappa = 5.  # mean-reversion speed
    theta = 0.08  # long term variance
    sigma = 0.3  # vol fo vol
    r = .05  # risk-free rate
    q = 0  # continuous dividends

    S, V, _ = EulerMilsteinSim(scheme, neg_var, num_sim, n_steps, rho, S0, V0, T, kappa, theta, sigma, r, q)
    # print(S[:, :10])
    fig, axs = plt.subplots(2)
    fig.suptitle('Stocks and Variance for the Heston Model')
    axs[0].plot(S[:, :10])
    axs[1].plot(V[:, :10])
    plt.show()
