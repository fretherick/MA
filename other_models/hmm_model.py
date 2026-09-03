import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from typing import Tuple  

class HiddenMarkovModel:
    def __init__(
            self, 
            k: int,
            initial_state_prob: np.ndarray,
            transition_matrix: np.ndarray,
            means: np.ndarray,
            variances: np.ndarray,
    ):
        """
        Initializes the Hidden Markov Model.

        Parameters:
            k (int): Number of states.
            initial_state_prob (array-like): Probability vector for the initial state (length k).
            transition_matrix (2D array-like): k x k matrix where each row sums to 1.
            means (array-like): Means for the Gaussian distribution in each state (length k).
            variances (array-like): Variances for the Gaussian distribution in each state (length k).
        """
        self.k = k
        self.initial_state_prob = np.array(initial_state_prob)
        self.transition_matrix = np.array(transition_matrix)
        self.means = np.array(means)
        self.variances = np.array(variances)

        # Validate inputs
        if self.initial_state_prob.shape != (k,):
            raise ValueError("initial_state_prob must be of shape (k,)")
        if not np.allclose(self.initial_state_prob.sum(), 1):
            raise ValueError("initial_state_prob must sum to 1")
        if self.transition_matrix.shape != (k, k):
            raise ValueError("transition_matrix must be of shape (k, k)")
        if not np.allclose(self.transition_matrix.sum(axis=1), 1):
            raise ValueError("transition_matrix rows must sum to 1")
        if self.means.shape != (k,):
            raise ValueError("means must be of shape (k,)")
        if self.variances.shape != (k,):
            raise ValueError("variances must be of shape (k,)")

    # def simulate(self, T: int, dt: int = 1) -> tuple[np.ndarray, np.ndarray]:
    #     """
    #     Simulate a time series of length T from the HMM.
    #     Parameters:
    #         T (int): The number of time steps.
    #         dt (int, optional): Time scale factor. Defaults to 1.
    #     Returns:
    #         hidden_states (list): The sequence of hidden states.
    #         observations (list): The sequence of observations sampled from the state-dependent Gaussians.
    #     """
    #     hidden_states = np.empty(T)
    #     log_returns = np.empty(T)
        
    #     from scipy.linalg import fractional_matrix_power
    #     # If dt > 1 we re-compute the dt-step transition matrix. For non-integer dt, matrix exponentiation is required.
    #     if dt == 20:
    #         trans_mat = self.transition_matrix
    #     else:
    #         trans_mat = np.linalg.matrix_power(self.transition_matrix, int(dt))

    #     # Initial state
    #     current_state = np.random.choice(self.k, p=self.initial_state_prob)
    #     for t in range(T):
    #         hidden_states[t] = current_state
    #         # Sample from the Gaussian distribution for the current state
    #         l_ret = np.random.normal(
    #             loc=dt * self.means[current_state],
    #             scale=np.sqrt(dt) * np.sqrt(self.variances[current_state])
    #         )
    #         log_returns[t] = l_ret
    #         # Transition to the next state
    #         current_state = np.random.choice(self.k, p=trans_mat[current_state])
    #     return hidden_states, log_returns
     # add this near the imports

    def simulate(self, T: int, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate a time series from the HMM.
        
        Parameters:
            T (int): Number of time steps to simulate.
            dt (float): Desired time step (e.g., 1 = daily if base is monthly).
            base_dt (float): Time scale of calibrated model (e.g., 20 for monthly model).
        
        Returns:
            hidden_states (np.ndarray): The sequence of hidden states.
            log_returns (np.ndarray): Simulated log-returns.
        """
        from scipy.linalg import fractional_matrix_power

        hidden_states = np.empty(T, dtype=int)
        log_returns = np.empty(T)

        # Time scaling factor
        scaling = dt / 20

        # Adjust transition matrix
        trans_mat = fractional_matrix_power(self.transition_matrix, scaling)

        # Initial state
        current_state = np.random.choice(self.k, p=self.initial_state_prob)
        for t in range(T):
            hidden_states[t] = current_state
            mean = scaling * self.means[current_state]
            std = np.sqrt(scaling) * np.sqrt(self.variances[current_state])
            log_returns[t] = np.random.normal(loc=mean, scale=std)
            current_state = np.random.choice(self.k, p=trans_mat[current_state])

        return hidden_states, log_returns

    def simulate_paths(self, nb_paths: int, T: int, dt: int = 1, more_states: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate more time series of length T from the HMM.
        Parameters:
            nb_paths (int): The number of different sequences/paths.
            T (int): The number of time steps.
            dt (int, optional): Time scale factor. Defaults to 1.
            more_states (bool, optional): Avoid returning sequences of 1 state only?
        Returns:
            hidden_states (np.ndarray): The sequence of hidden states.
            observations (np.ndarray): The sequence of Gaussian log-returns.
        """
        # Use function simulate to generate more samples
        hidden_states = np.full((nb_paths, T), fill_value=np.nan)
        log_returns = np.full((nb_paths, T), fill_value=np.nan)
        i = 0
        while i < nb_paths:
            h_states, log_rets = self.simulate(T, dt)
            # In case I want to avoid sequences made by 1 state only
            if more_states and len(set(h_states)) == 1:
                continue
            hidden_states[i] = np.array(h_states)
            log_returns[i] = np.array(log_rets)
            i += 1
        return hidden_states, log_returns


    @staticmethod
    def plot_simulation(hidden_states: np.ndarray, observations: np.ndarray):
        prices = np.exp(np.cumsum(observations))
        time = np.arange(len(observations))

        # Function to linearly scale hidden states to fit a given data range
        def scale_states(states, data_min, data_max):
            hs_min, hs_max = states.min(), states.max()
            # Avoid division by zero if hs_min == hs_max
            if hs_min == hs_max:
                return np.full_like(states, (data_min + data_max) / 2)
            return (states - hs_min) / (hs_max - hs_min) * (data_max - data_min) + data_min

        # Scale hidden states for the returns plot
        ret_min, ret_max = observations.min(), observations.max()
        hs_returns_scaled = scale_states(hidden_states, ret_min, ret_max)
        # Scale hidden states for the cumulative performance plot
        cp_min, cp_max = prices.min(), prices.max()
        hs_p_scaled = scale_states(hidden_states, cp_min, cp_max)

        # Create a subplot with 2 rows sharing the same x-axis
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=("Simulated Returns", "Prices"),
            vertical_spacing=0.1,
        )
        # Overlay scaled hidden states on the returns plot
        fig.add_trace(
            go.Scatter(x=time, y=hs_returns_scaled, mode='lines', name='Hidden States (scaled)',
                       line=dict(color='red', dash='dash', width=0.9)),
            row=1, col=1
        )
        # Plot the returns
        fig.add_trace(
            go.Scatter(x=time, y=observations, mode='lines', name='Returns', line=dict(color='blue')),
            row=1, col=1
        )

        # Overlay scaled hidden states on the cumulative performance plot
        fig.add_trace(
            go.Scatter(x=time, y=hs_p_scaled, mode='lines', name='Hidden States (scaled)',
                       line=dict(color='red', dash='dash', width=0.9)),
            row=2, col=1
        )
        # Plot the cumulative performance
        fig.add_trace(
            go.Scatter(x=time, y=prices, mode='lines', name='Prices',
                       line=dict(color='orange')),
            row=2, col=1,
        )

        # Update layout and axis labels
        fig.update_layout(
            height=600,
            width=800,
            title_text="HMM Simulation with Hidden States",
            showlegend=True,
        )
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Returns", row=1, col=1)
        fig.update_yaxes(title_text="Prices", row=2, col=1)
        fig.show()


if __name__ == "__main__":
    np.random.seed(1)
    # Define parameters for a 3-state HMM
    k = 3
    initial_state_prob = np.array([0.65, 0.2, 0.15])
    transition_matrix = np.array([
        [0.85, 0.1, 0.05],
        [0.12, 0.76, 0.12],
        [0.18, 0.02, 0.80],
    ])
    means = np.array([0.0, 0.012, -0.017])
    variances = np.array([0.0001, 0.0002, 0.003])

    hmm = HiddenMarkovModel(k, initial_state_prob, transition_matrix, means, variances)
    hidden_states, log_rets = hmm.simulate(T=1000)
    hmm.plot_simulation(hidden_states, log_rets)

    hs, lrets = hmm.simulate_paths(nb_paths=3, T=250, more_states=True)

    # 2 states only
    k = 2
    initial_state_prob = np.array([0.95, 0.05])
    transition_matrix = np.array([
        [0.98, 0.02],
        [0.12, 0.88],
    ])
    means = np.array([0.0, 0.01])
    variances = np.array([0.0001, 0.0002])
    hmm = HiddenMarkovModel(k, initial_state_prob, transition_matrix, means, variances)
    hs, lrets = hmm.simulate_paths(3, 100, more_states=True)

    