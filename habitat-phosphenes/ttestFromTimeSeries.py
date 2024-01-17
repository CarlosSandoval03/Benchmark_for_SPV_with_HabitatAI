from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import scipy.stats as stats
import seaborn as sns
import os


def parse_tensorboard(base_dir, file_paths, scalars):
    """
    Returns a dictionary of pandas dataframes. 1 dataFrame per metric of interest
    that will append the results from multiple paths (tensorbord files)
    """
    all_data = {scalar: pd.DataFrame() for scalar in scalars}

    for file_path in file_paths:
        full_path = os.path.join(base_dir, file_path)
        ea = event_accumulator.EventAccumulator(
            full_path,
            size_guidance={event_accumulator.SCALARS: 0},
        )
        _absorb_print = ea.Reload()
        # make sure the scalars are in the event accumulator tags
        assert all(
            s in ea.Tags()["scalars"] for s in scalars
        ), f"Some metrics were not found in the event accumulator"

        for scalar in scalars:
            scalar_data = pd.DataFrame(ea.Scalars(scalar))
            scalar_data['source'] = file_path
            all_data[scalar] = pd.concat([all_data[scalar], scalar_data], ignore_index=True)

    return all_data


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


def plot_metrics(data, file_labels, metrics_of_interest, step_range=None, save_figures=False, figure_names=None, figure_titles=None, smoothing_method='ema', smooth_factor=0.0):
    """
    Plots the time series.
        - One figure per metric of interest in metrics_of_interest
        - Each figure has as many time series as tensorbord files provided and assign the file_labels in that order
        - The figures plot the region of variability of +- std
        - The figures will be saved if save_figures=True and figure_titles and figure_names are provided
    """
    show_grid = False
    sns.set(style="whitegrid" if show_grid else "white")
    for i, metric in enumerate(metrics_of_interest):
        plt.figure()
        y_min, y_max = float('inf'), float('-inf')  # Initialize to extreme values
        for file_path, df_group in data[metric].groupby('source'):
            label_index = file_paths.index(file_path)
            label = file_labels[label_index] if file_labels and label_index < len(file_labels) else file_path

            if step_range and len(step_range) == 2 and step_range[1] <= df_group['step'].max():
                df_group = df_group[(df_group['step'] >= step_range[0]) & (df_group['step'] <= step_range[1])].copy()
            else:
                df_group = df_group.copy()

            if smooth_factor > 0:
                df_group['smoothed'] = smooth_data(df_group['value'], method=smoothing_method,
                                                   smooth_factor=smooth_factor)
                mean_line = df_group['smoothed']
            else:
                mean_line = df_group['value']

            # Calculate standard deviation for shadow
            std_dev = df_group['value'].rolling(window=10, min_periods=1).std()

            current_color = sns.color_palette()[label_index]
            plt.fill_between(df_group['step'], mean_line - std_dev, mean_line + std_dev,
                             color=current_color, alpha=0.15)
            sns.lineplot(data=df_group, x='step', y=mean_line, label=label, linewidth=2.5, color=current_color)

            y_min = min(y_min, mean_line.min())
            y_max = max(y_max, mean_line.max())

        y_margin = (y_max - y_min) * 0.05
        plt.ylim(y_min - y_margin, y_max + y_margin)  # Set y-axis limits

        plt.tight_layout()
        plt.title(figure_titles[i] if figure_titles and i < len(figure_titles) else metric)
        plt.xlabel('Step')
        plt.ylabel(metric.split('/')[-1].replace('_', ' '))
        plt.legend()

        if save_figures and figure_names and i < len(figure_names):
            plt.savefig(figure_names[i], dpi=300, bbox_inches='tight')
        plt.show()

def perform_ttest_and_plot(data, scalar, labelsBoxplot, save_path, figur_title):
    # Ensure the labelsBoxplot array matches the number of unique sources
    assert len(labelsBoxplot) == len(data['source'].unique()), "Labels array size must match the number of unique sources"

    # Replace source names in the DataFrame with provided labels
    label_dict = {source: label for source, label in zip(data['source'].unique(), labelsBoxplot)}
    data['label'] = data['source'].map(label_dict)

    # Filter the data for the last 100 entries of each file
    filtered_data = data.groupby('label').apply(lambda x: x.tail(100))

    # Perform t-test between each pair of labels and store results
    t_test_results = []
    for i in range(len(labelsBoxplot)):
        for j in range(i + 1, len(labelsBoxplot)):
            data1 = filtered_data[filtered_data['label'] == labelsBoxplot[i]]["value"]
            data2 = filtered_data[filtered_data['label'] == labelsBoxplot[j]]["value"]
            t_stat, p_val = stats.ttest_ind(data1, data2)
            t_test_results.append((labelsBoxplot[i], labelsBoxplot[j], t_stat, p_val))

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='label', y="value", data=filtered_data)
    plt.title(f'Boxplot for {figur_title}', fontsize=20)
    plt.ylabel(figur_title, fontsize=16)
    plt.xlabel("Simulation's ID", fontsize=16)


    # Annotate with t-test results
    for i, (label1, label2, t_stat, p_val) in enumerate(t_test_results):
        text_color = 'darkgreen' if p_val < 0.05 else 'darkred'
        plt.text(0.5, 0.95 - 0.05 * i, f'{label1} vs {label2}: t={t_stat:.2e}, p={p_val:.2e}',
                 horizontalalignment='center', color=text_color, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.2), fontsize=16)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    base_dir = "/scratch/big/home/carsan/Data/phosphenes/habitat/tb"

    folder_paths = [
        "train_CVSegmentation/train_NPT_RGB_backgroundGrayEdgesPhosphenes_2.8Msteps_8Envs_MaxEpsStep800",
        "train_CVSegmentation/train_NPT_RGB_backgroundGrayEdgesPhosphenes_2.8Msteps_8Envs_MaxEpsStep800_run2",
        # "train_CVSegmentation/train_NPT_GPSRGB_backgroundGrayEdgesPhosphenes_2.8Msteps_8Envs_MaxEpsStep800",
        # "train_Baselines/train_NPT_GPS_Original_2.8Msteps",
        # "train_Baselines/train_NPT_GPSDepthRGB_Original_2.8Msteps_8Envs",
        # "train_Baselines/train_NPT_GPSDepth_Original_2.8Msteps_8Envs",
        # "train_Baselines/train_NPT_GPSRGB_Original_2.8Msteps_8Envs"
    ]
    # Automatically detect the log file in each folder
    file_paths = []
    for folder in folder_paths:
        full_folder_path = os.path.join(base_dir, folder)
        log_files = [f for f in os.listdir(full_folder_path) if f.startswith("events.out.tfevents")]
        if log_files:
            file_paths.append(os.path.join(folder, log_files[0]))  # Append the first log file found

    file_labels = [
        "kernelSize=10",
        "kernelSize=5",
        # "GPSRGB_kernelSize=10",
        # "GPS",
        # "GPSRGBDepth",
        # "GPSDepth",
        # "GPSRGB"
    ]  # For the time series, 1 per file provided
    figure_names = [
        "./Boxplots/evaluationMetrics/dtg_bp.png",
        "./Boxplots/evaluationMetrics/spl_bp.png",
        "./Boxplots/evaluationMetrics/reward_bp.png",
        "./Boxplots/evaluationMetrics/de_bp.png"
    ]  # 1 per metric of interest
    metrics_of_interest = ["metrics/distance_to_goal",
                           "metrics/spl",
                           "reward",
                           # "metrics/success"
                           # "learner/recon_loss",
    ]  # eval_metrics/ or metrics/ or learner/recon_loss
    figure_titles = ["Distance to Goal", "SPL", "Reward", "ReconstructionLoss"]  # 1 per figure (metric of interest)

    step_range = [0]  # If only 1 value then it plots everything, if two values it plots that range
    save_figures = True
    smoothing_method = 'ema'  # ema, gaussian, or savgol
    smooth_factor = 0.01  # Smoothing factor (1 for no smoothing, opposite to TB)

    data = parse_tensorboard(base_dir, file_paths, metrics_of_interest)

    for i, scalar in enumerate(metrics_of_interest):
        figure_titles = ["Distance to Goal", "SPL", "Reward", "ReconstructionLoss"]  # 1 per figure (metric of interest)
        perform_ttest_and_plot(data[scalar], scalar, file_labels, figure_names[i], figure_titles[i])
