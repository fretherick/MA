import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
############ resample_data() for CLOSE AND VOLUME ###########
def resample_data(data, resample_rule='5T', start_date=None, end_offset_years=2, price_column='close', volume_column='volume'):
    # Convert the 'datetime' column to datetime format
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    # Set 'datetime' as the index
    data.set_index('datetime', inplace=True)
    
    # Define the end date as the given offset from the start date
    end_date = pd.to_datetime(start_date) + pd.DateOffset(years=end_offset_years)
    
    # Filter data for the defined period starting from start_date
    data = data.loc[start_date:end_date]
    
    # Resample and aggregate
    resampled_data = data.resample(resample_rule).agg({
        price_column: 'last',   # Last price in the resample window
        volume_column: 'sum',   # Sum of volume in the resample window
    })
    
    # Handle missing data by forward filling
    resampled_data.ffill(inplace=True)
    
    return resampled_data
######################### END ###################################

################## PLOTTER FUNCTION #############################
def plot_btc_price(predictions, btc_prices, output_path, title="BTC Price Colored by Predictions"):
    """
    Plots BTC prices with color-coded segments based on predictions.

    Args:
        predictions (np.ndarray): Predicted cluster labels or hidden states.
        btc_prices (np.ndarray): BTC price data to plot.
        output_path (str): Path to save the output image.
        title (str): Title for the plot.
    """

    # Define the custom colormap: green for one state, red for the other
    custom_cmap = ListedColormap(['green', 'red'])

    # Create figure
    fig, axs = plt.subplots(1, 1, figsize=(14, 10))  # Increased figure size

    # Plot BTC prices in segments according to predictions
    for i in range(len(predictions - 1)):
        axs.plot([i, i+1], [btc_prices[i], btc_prices[i+1]], 
                 color=custom_cmap(predictions[i]), alpha=0.7)

    # Create a custom legend
    legend_elements = [Line2D([0], [0], color='green', lw=4, label='State 1'),
                       Line2D([0], [0], color='red', lw=4, label='State 2')]

    # Add the legend to the plot
    axs.legend(handles=legend_elements, loc='upper left', fontsize=14)

    # Set the title and labels
    axs.set_title(title, fontsize=18)
    axs.set_xlabel('Index', fontsize=16)
    axs.set_ylabel('BTC Price', fontsize=16)
    axs.grid(True)  # Add grid

    # Improve layout and add padding
    plt.tight_layout(pad=3.0)

    # Save the figure
    plt.savefig(output_path)

    # Show the plot
    plt.show()
######################### END PLOTTER FUNCTION###################################