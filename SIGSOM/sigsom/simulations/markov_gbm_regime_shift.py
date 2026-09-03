import numpy as np
import matplotlib.pyplot as plt
# Switch according to the Markov Chain
def markov_regime_switch(current_state, transition_matrix):
    """
    Given a current state and a transition matrix, return the next state based on probabilities.
    """
    return np.random.choice(len(transition_matrix), p=transition_matrix[current_state])

# Generate the GBM
def Geom_Brow_sim_with_markov(num_sim, n_steps, T, S0, intervals, transition_matrix, max_interval_steps=400):
    # time step size
    dt = T / n_steps
    # save the sim with shape [n_steps, num_sim] to exclude the initial value
    S = np.zeros((n_steps, num_sim))
    S[0, :] = S0
    
    # Initialize regime tracking arrays
    regimes = np.zeros(n_steps, dtype=int)
    regime_lengths = [] 
    # Start t
    current_t = 0
    # Randomly Choose start
    current_state = np.random.choice([0, 1], p=[0.5, 0.5])
    
    while current_t < n_steps:
        # Get the parameters for the current regime
        mu, sigma = intervals[current_state][:2]
        
        # Generate the number of steps in the current regime
        interval_steps = np.random.randint(100, max_interval_steps)  # Random interval length
        interval_steps = min(interval_steps, n_steps - current_t)  # Ensure we don't exceed total steps
        
        # Store the regime and its length
        regime_lengths.append(interval_steps)
        regimes[current_t:current_t + interval_steps] = current_state
        
        # Simulate for this regime
        for i in range(num_sim):
            for t in range(current_t + 1, current_t + interval_steps + 1):
                if t >= n_steps:  # Prevent out-of-bounds indexing
                    break
                Z = np.random.standard_normal()
                S[t, i] = S[t-1, i] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

        current_t += interval_steps

        # Switch to the next regime based on the Markov process
        current_state = markov_regime_switch(current_state, transition_matrix)

    return S, regimes, regime_lengths
# use n_step 12000, and widnow length 600



if __name__ == '__main__':
    # Parameters
    n_steps = 24000    
    T = 1         
    S0 = 100          
    num_sim = 1

    # Intervals
    intervals = [
        (0.2, 0.05),   # Regime 0
        (-0.2, 0.05),   # Regime 1
    ]

    # Regimes
    transition_matrix = np.array([
        [0.85, 0.15],
        [0.15, 0.85],
    ])



    S, regimes, regime_lengths = Geom_Brow_sim_with_markov(num_sim, n_steps, T, S0, intervals, transition_matrix)

    # Plot the simulated asset price path
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(n_steps), S[:, 0], label='Asset Price')
    plt.title('Geometric Brownian Motion with Markov Regime Switching')
    plt.xlabel('Time Steps')
    plt.ylabel('Asset Price')
    plt.legend()
    plt.show()

    # Plot the regime over time using a step plot
    plt.figure(figsize=(10, 3))
    plt.step(np.arange(n_steps), regimes, where='post', label='Regime')
    plt.title('Regime over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Regime')
    plt.yticks([0, 1])
    plt.legend()
    plt.show()