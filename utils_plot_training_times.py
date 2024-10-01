import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from datetime import datetime, timedelta
from matplotlib import font_manager
from matplotlib.cm import get_cmap

def timestamp_to_seconds(duration_str):
    time_object = datetime.strptime(duration_str, '%H:%M:%S')
    total_seconds = time_object.hour * 3600 + time_object.minute * 60 + time_object.second
    return total_seconds

def seconds_to_timestamp(total_seconds):
    # Convert total_seconds to hours, minutes, and seconds
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    # Format the string with two digits precision for seconds
    duration_str = "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))
    return duration_str

base_folder = './' # '//wsl.localhost/Ubuntu/home/marce/git/auto-ml-comparison-2024/'

font_dirs = [f'{base_folder}/fonts']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
    prop = font_manager.FontProperties(fname=font_file)
    print(font_file, prop.get_name())

plt.rcParams['font.family'] = 'CMU Serif'

datasets = {
    'binary': [37, 44, 1462, 1479, 1510],
    'multiclass': [23, 181, 1466, 40691, 40975],
    'multilabel_native': [41465, 41468, 41470, 41471, 41473],
    'multilabel_powerset': ['41465ps', '41468ps', '41470ps', '41471ps', '41473ps']
}

frameworks = ['4intelligence', 'AutoGluon', 'AutoKeras', 'Auto-PyTorch', 'AutoSklearn', 'EvalML', 'FEDOT',
              'FLAML', 'GAMA', 'H2O', 'LightAutoML', 'mljar-supervised', 'naiveautoml', 'PyCaret', 'TPOT', 'VowpalWabbit']

for scenario, dataset_refs in datasets.items():

    df, data = pd.read_excel(f'{base_folder}/results/{scenario}.xlsx', sheet_name='training_time'), {}
    for dataset_ref in dataset_refs:
        data[dataset_ref] = df[dataset_ref].fillna('00:00:00 (00:00:00 Â± 00:00:00)')

    pattern = r'(\d+:\d+:\d+)\s+\((\d+:\d+:\d+)\s*Â±\s*(\d+:\d+:\d+)\)'

    # Extract min, mean, and standard deviation values
    min_vals = np.array([[re.match(pattern, value).group(1) for value in data[key]] for key in data.keys()])
    for i, min_val in enumerate(min_vals):
        for j, duration_str in enumerate(min_val):
            min_vals[i][j] = timestamp_to_seconds(duration_str)
    min_vals = min_vals.astype(int)
    print(min_vals)

    mean_vals = np.array([[re.match(pattern, value).group(2) for value in data[key]] for key in data.keys()])
    for i, mean_val in enumerate(mean_vals):
        for j, duration_str in enumerate(mean_val):
            mean_vals[i][j] = timestamp_to_seconds(duration_str)
    mean_vals = mean_vals.astype(int)
    print(mean_vals)

    stdev_vals = np.array([[re.match(pattern, value).group(3) for value in data[key]] for key in data.keys()])
    for i, stdev_val in enumerate(stdev_vals):
        for j, duration_str in enumerate(stdev_val):
            stdev_vals[i][j] = timestamp_to_seconds(duration_str)
    stdev_vals = stdev_vals.astype(int)
    print(stdev_vals)

    upper_limit = np.ceil((np.max([min_vals, mean_vals, stdev_vals]) / 300) * 300)
    y_ticks = np.linspace(0, upper_limit, 6)
    print('upper_limit = ', upper_limit, 'y_ticks = ', y_ticks)

    # Set up positions for each bar
    positions = np.arange(len(data.keys())) / 1.25

    # Get colormap
    cmap = get_cmap('tab20') # Paired
    # Plotting
    fig, ax = plt.subplots(figsize=(22, 7))
    ax.yaxis.grid(True, linestyle='dashed', linewidth=0.5, alpha=0.7)  # Enable y-grid with customized properties

    for i, framework in enumerate(frameworks):
        x_vals = positions + i * 0.04  # Adjust the width between bars
        color = cmap(i / len(frameworks))  # Color based on framework index
        ax.bar(x_vals, mean_vals[:, i], width=0.04, label=framework, color=color)
        ax.errorbar(x_vals, mean_vals[:, i], yerr=stdev_vals[:, i], fmt='none', capsize=5, color='dimgray')
        # Check if min_vals is non-zero before plotting the marker
        non_zero_min_vals = [val if val != 0 else np.nan for val in min_vals[:, i]]
        ax.scatter(x_vals, non_zero_min_vals, marker='*', color='red')

    ax.set_xticks(positions + 0.04 * (len(frameworks) - 1) / 2)  # Set ticks at the center of each group
    ax.set_xticklabels([str(x).replace('ps', '') for x in data.keys()], fontsize=20)  # Increase font size for x-axis labels
    ax.set_ylim(0, np.max(y_ticks) + 120)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([seconds_to_timestamp(y) for y in y_ticks], fontsize=20)  # Increase font size for x-axis labels
    ax.set_xlabel(r'Dataset', fontsize=24)  # Increase font size for x-axis label
    ax.set_ylabel(r'Training Time', fontsize=24)  # Increase font size for y-axis label
    ax.legend(loc='best', bbox_to_anchor=(1, 1), fontsize=20)  # Increase font size for legend
    # ax.set_title(r'Training Time (Min, Mean, Stdev) for each Framework and Dataset', fontsize=24)  # Increase font size for title

    plt.tight_layout()
    plt.savefig(f'{base_folder}/results/training_time_{scenario}.png')
    plt.show()
