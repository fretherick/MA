import numpy as np
import matplotlib.pyplot as plt

# Function to simulate Geometric Brownian Motion with changing mu
def Geom_Brow_sim_with_intervals(num_sim, n_steps, T, S0, intervals):
    # time step size
    dt = T / n_steps
    # save the sim
    S = np.zeros((n_steps + 1, num_sim))
    S[0, :] = S0

    current_t = 0
    for mu, sigma, interval_steps in intervals:
        for i in range(num_sim):
            for t in range(current_t + 1, current_t + interval_steps + 1):
                Z = np.random.randn(1)
                S[t, i] = S[t-1, i] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        current_t += interval_steps

    return S

if __name__ == '__main__':
    # Parameters
    num_sim = 10     
    n_steps = 100    
    T = 1.0           
    S0 = 100          

    
    intervals = [
        (0.2, 0.07, 20),
        (-0.4, 0.1, 20),  
        (0.1, 0.05, 20),   
        (-0.2, 0.09, 20),  
        (0.1, 0.05, 20)    
    ]

    # Run simulation 10 times
    for j in range(1):
        simulated_paths = Geom_Brow_sim_with_intervals(num_sim, n_steps, T, S0, intervals)

        # Display some simulated paths
        plt.figure(figsize=(10, 6))
        for i in range(min(num_sim, 10)):
            plt.plot(simulated_paths[:, i])
        plt.title(f'Geometric Brownian Motion Simulation {j+1}')
        plt.xlabel('Time Steps')
        plt.ylabel('Asset Price')
        plt.show()

