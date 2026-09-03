import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
import torch

def generate_arma(ar_params, ma_params, n_samples, n_length, constant=0.5, random_seed=None):
    """
    Generate an ARMA process sample.

    Parameters:
      ar_params (array-like): The AR coefficients (not including the zero lag).
      ma_params (array-like): The MA coefficients (not including the zero lag).
      n_samples (int): Number of independent time series to generate.
      n_length (int): Length of each time series.
      constant (float): A constant to add to the generated series.
      random_seed (int or None): Random seed for reproducibility.

    Returns:
      np.ndarray: A generated ARMA process sample of shape (n_samples, n_length).
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    # Build the AR and MA arrays for the ArmaProcess
    ar = np.r_[1, -np.asarray(ar_params)]
    ma = np.r_[1, np.asarray(ma_params)]
    # Create the ARMA process and generate the samples
    process = ArmaProcess(ar, ma)
    samples = process.generate_sample(nsample=(n_samples, n_length))
    samples += constant  # add the constant
    return samples

if __name__ == "__main__":
    # Parameters for the simulation
    np.random.seed(42)
    n_samples = 500
    n_length = 500
    constant = 0.5
    arma_process_0 = generate_arma([0.4], [0.5], n_samples, n_length, constant, random_seed=42)

    plt.figure(figsize=(20, 4))
    plt.subplot(1, 1, 1)
    plt.plot(arma_process_0[0])
    plt.title('ARMA Class 0')
    plt.show()