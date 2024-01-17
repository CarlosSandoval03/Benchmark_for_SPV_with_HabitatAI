from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import seaborn as sns
import os


def parse_json(base_dir, file_path):
    """
    Returns a dictionary of pandas dataframes. 1 dataFrame per metric of interest
    that will append the results from multiple paths (tensorbord files)
    """
    full_path = os.path.join(base_dir, file_path)
    with open(full_path, 'r') as file:
        data = json.load(file)

    x_values = [item[1] for item in data]
    y_values = [item[2] for item in data]

    # Create the DataFrame
    df = pd.DataFrame({
        'x': x_values,
        'y': y_values
    })

    # Calculate the Exponential Moving Average (EMA) for smoothing
    # df['y_ema'] = df['y'].ewm(alpha=smooth_factor).mean()
    # Calculate the standard deviation for the shaded area
    # std_dev = df['y'].ewm(span=20, adjust=False).std()

    return df

def smooth_data(data, method='ema', smooth_factor=0.0, window_length=5, polyorder=2):
    """
    Smoothing the time series to have a clearer representation of the tendency of the data
    According to internet, the method used by default in tensorbord is Exponential Moving Average
    """
    if method == 'ema':  # Exponential Moving Average
        if smooth_factor <= 0 or smooth_factor >= 1:
            return data
        return data.ewm(alpha=smooth_factor).mean()
    elif method == 'gaussian':  # Gaussian Smoothing
        return gaussian_filter1d(data, sigma=smooth_factor)
    elif method == 'savgol':  # Savitzky-Golay Filter
        if window_length < 3 or window_length % 2 == 0:
            window_length = 5  # default to 5 if invalid
        return savgol_filter(data, window_length, polyorder)
    else:
        return data

def plot_metrics(base_dir, file_paths, file_labels, save_figures=False, figure_names=None, figure_titles=None, smoothing_method='ema', smooth_factor=0.0):
    """
    Plots the time series.
        - One figure per metric of interest in metrics_of_interest
        - Each figure has as many time series as tensorbord files provided and assign the file_labels in that order
        - The figures plot the region of variability of +- std
        - The figures will be saved if save_figures=True and figure_titles and figure_names are provided
    """
    show_grid = False
    sns.set(style="whitegrid" if show_grid else "white")

    plt.figure()
    y_min, y_max = float('inf'), float('-inf')  # Initialize to extreme values
    for file_path in file_paths:
        label_index = file_paths.index(file_path)
        label = file_labels[label_index] if file_labels and label_index < len(file_labels) else file_path

        data = parse_json(base_dir, file_path)

        if smooth_factor > 0:
            data['y_ema'] = smooth_data(data['y'], method=smoothing_method, smooth_factor=smooth_factor)
            mean_line = data['y_ema']
        else:
            mean_line = data['y_ema']

        # Calculate standard deviation for shadow
        std_dev = data['y'].rolling(window=10, min_periods=1).std()

        plt.fill_between(data['x'], mean_line - std_dev, mean_line + std_dev,
                         color=sns.color_palette()[label_index], alpha=0.15)
        sns.lineplot(data=data, x='x', y=mean_line, label=label, linewidth=2.5)

        y_min = min(y_min, mean_line.min())
        y_max = max(y_max, mean_line.max())

    y_margin = (y_max - y_min) * 0.05
    plt.ylim(y_min - y_margin, y_max + y_margin)  # Set y-axis limits

    plt.title(figure_titles)
    plt.xlabel('Step')
    plt.ylabel(figure_titles)
    plt.legend()

    if save_figures:
        plt.savefig(figure_names, dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    base_dir = "/scratch/big/home/carsan/Internship/PyCharm_projects/habitat_2.3/habitat-phosphenes/simulationDataJSON"

    file_paths = [
        "train_NPT_Depth_3transf.json",
        "train_NPT_RGB_Original_2.8Msteps_dtg.json",
    ]

    file_labels = [
        "Depth_Phos",
        "Original",
    ] # For the time series, 1 per file provided

    figure_name = "distance_to_goal_json.png"
    figure_title = "Distance to Goal"

    save_figures = True
    smoothing_method = 'ema'  # ema, gaussian, or savgol
    smooth_factor = 0.01  # Smoothing factor (1 for no smoothing, opposite to TB)

    plot_metrics(base_dir, file_paths, file_labels, save_figures, figure_name, figure_title, smoothing_method, smooth_factor)

    print("END")
