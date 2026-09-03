import numpy as np
import matplotlib.pyplot as plt

# @nb.jit
def Geom_Brow_sim(num_sim, n_steps, T, S0, mu, sigma):
    # time step size
    dt = T / n_steps
    # save the sim
    S = np.zeros((n_steps + 1, num_sim))
    S[0, :] = S0

    for i in range(num_sim):
        for t in range(1, n_steps + 1):
            # Z = np.random.randn(1)
            Z = np.random.standard_normal()
            S[t, i] = S[t-1, i] * np.exp( (mu - 0.5 * sigma**2) * dt + np.sqrt(dt) * sigma * Z)

    return S
if __name__ == '__main__':
    # Parameters
    num_sim = 100     # Number of simulation paths
    n_steps = 100       # Number of time steps )
    T = 1.0             # Time horizon
    S0 = 100            # Initial asset price
    mu = 0.05           # Drift 
    sigma = 0.2         # Volatility 

    # Run simulation
    simulated_paths = Geom_Brow_sim(num_sim, n_steps, T, S0, mu, sigma)

    # Display some simulated paths
    plt.figure(figsize=(10,6))
    for i in range(min(num_sim, 10)):
        plt.plot(simulated_paths[:, i])
    plt.title('Geometric Brownian Motion Simulation')
    plt.xlabel('Time Steps')
    plt.ylabel('Asset Price')
    plt.show()
