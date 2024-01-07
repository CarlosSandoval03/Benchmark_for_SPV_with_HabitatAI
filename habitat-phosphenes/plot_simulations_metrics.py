from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import seaborn as sns
import os


def parse_tensorboard(base_dir, file_paths, scalars):
    """
    Returns a dictionary of pandas dataframes. 1 dataFramper etric of interest
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
        ), "Some metrics were not found in the event accumulator"

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

            if step_range and len(step_range) == 2:
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

            plt.fill_between(df_group['step'], mean_line - std_dev, mean_line + std_dev,
                             color=sns.color_palette()[label_index], alpha=0.15)
            sns.lineplot(data=df_group, x='step', y=mean_line, label=label, linewidth=2.5)

            y_min = min(y_min, mean_line.min())
            y_max = max(y_max, mean_line.max())

        y_margin = (y_max - y_min) * 0.05
        plt.ylim(y_min - y_margin, y_max + y_margin)  # Set y-axis limits

        plt.title(figure_titles[i] if figure_titles and i < len(figure_titles) else metric)
        plt.xlabel('Step')
        plt.ylabel(metric.split('/')[-1].replace('_', ' '))
        plt.legend()

        if save_figures and figure_names and i < len(figure_names):
            plt.savefig(figure_names[i], dpi=300, bbox_inches='tight')
        plt.show()



if __name__ == "__main__":
    base_dir = "/scratch/big/home/carsan/Data/phosphenes/habitat/tb"

    folder_paths = [
        # "train_Baselines/train_NPT_GPS_3transf",
        # "train_Baselines/train_NPT_GPS_Original",
        # "train_Baselines/train_NPT_RGB_3transf_2.8Msteps",
        "train_E2E/train_NPT_GPSRGB_E2E_2.8Msteps_6Envs_800spe_xavierWeights_lr=2.5e-4_32spu_weightLossPPO=0.6",
        "train_E2E/train_NPT_RGB_E2E_2.8Msteps_2Envs_800spe_xavierWeights_lr=e-4_32spu_weightLossPPO=0.6_decoderWithRelu",
        # "train_CVSegmentation/train_NPT_RGB_backgroundIttiGrayEdgesPhosphenes_2.8Msteps_8Envs_MaxEpsStep800",
        # "train_CVSegmentation/train_NPT_RGB_ittiGrayEdgesPhosphenes_2.8Msteps_8Envs_MaxEpsStep800"
    ]
    # Automatically detect the log file in each folder
    file_paths = []
    for folder in folder_paths:
        full_folder_path = os.path.join(base_dir, folder)
        log_files = [f for f in os.listdir(full_folder_path) if f.startswith("events.out.tfevents")]
        if log_files:
            file_paths.append(os.path.join(folder, log_files[0]))  # Append the first log file found

    file_labels = [
        # "GPS_phos",
        # "GPS",
        # "RGB_phos",
        "GPSRGB_E2E",
        "RGB_E2E",
        # "RGB_BGSCPhos",
        # "RGB_Itti"
    ] # For the time series, 1 per file provided
    figure_names = ["distance_to_goal.png", "spl.png", "success.png"] # 1 per metric of interest
    metrics_of_interest = ["metrics/distance_to_goal", "metrics/spl", "metrics/success"] # eval_metrics/ or metrics/ or learner/recon_loss
    figure_titles = ["Distance to Goal", "SPL", "Success"] # 1 per figure (metric of interest)

    step_range = [0] # If only 1 value then it plots everything, if two values it plots that range
    save_figures = True
    smoothing_method = 'ema'  # ema, gaussian, or savgol
    smooth_factor = 1  # Smoothing factor (1 for no smoothing, opposite to TB)

    data = parse_tensorboard(base_dir, file_paths, metrics_of_interest)
    plot_metrics(data, file_labels, metrics_of_interest, step_range, save_figures, figure_names, figure_titles, smoothing_method, smooth_factor)

    print("END")
