import pandas as pd
from matplotlib import pyplot as plt


def plot_time_series_with_background(y, labels, title="", label_colors=None):
    """
    Plots a time series with background color depending on another time series with labels.

    Parameters:
    - y: pd.Series, time series values (index should be datetime)
    - labels: pd.Series, categorical or numerical labels (same index as y)
    - label_colors: dict, optional, mapping of labels to colors (e.g., {0: 'red', 1: 'green'})
    """
    if not isinstance(y, pd.Series) or not isinstance(labels, pd.Series):
        raise ValueError("Both y and labels must be pandas Series with a datetime index.")

    if not y.index.equals(labels.index):
        raise ValueError("y and labels must have the same index.")

    if title == "":
        title = "Time Series with Background Labels"

    unique_labels = labels.unique()

    if label_colors is None:
        cmap = plt.cm.get_cmap("tab10", len(unique_labels))
        label_colors = {label: cmap(i) for i, label in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(12, 6))

    # Fill background color based on labels
    for start, end, label in zip(labels.index[:-1], labels.index[1:], labels.iloc[:-1]):
        ax.axvspan(start, end, color=label_colors[label], alpha=0.3)

    # Plot the time series
    ax.plot(y.index, y.values, color='black', linewidth=2, label='Time Series')

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=label_colors[label]) for label in unique_labels]
    ax.legend(handles, unique_labels, title="Labels")

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(title)
    plt.show()



