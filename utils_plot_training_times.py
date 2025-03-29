import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import warnings
from datetime import datetime, timedelta
from matplotlib import colormaps, font_manager
from utils_consolidator import PLOT_COLORS

warnings.filterwarnings("ignore", message=".*Matplotlib is currently using agg.*")
warnings.filterwarnings("ignore", message=".*FixedFormatter.*")

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

base_folder = './'  # '//wsl.localhost/Ubuntu/home/marce/git/auto-ml-comparison-2024/'

# sudo apt-get install fonts-cmu
font_dirs = [f'{base_folder}/fonts']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
    prop = font_manager.FontProperties(fname=font_file)
    print(font_file, prop.get_name())

plt.rcParams['font.family'] = ['CMU Serif', 'DejaVu Sans']
plt.rcParams['text.usetex'] = False

datasets = {
    'binary': [31, 37, 44, 1462, 1479, 1510, 40945],
    'multiclass': [23, 36, 54, 181, 1466, 40691, 40975],
    'multilabel_native': [285, 41464, 41465, 41468, 41470, 41471, 41473],
    'multilabel_powerset': ['285ps', '41464ps', '41465ps', '41468ps', '41470ps', '41471ps', '41473ps']
}

frameworks = [
    '4intelligence', 'AutoGluon', 'AutoKeras', 'Auto-PyTorch', 'AutoSklearn', 'EvalML', 'FEDOT',
    'FLAML', 'GAMA', 'H2O', 'LightAutoML', 'Lightwood', 'mljar-supervised', 'NaiveAutoML', 'PyCaret', 'TPOT'
]

y_ticks = {
    'binary': 120 * np.array([0, 1, 2, 3, 4]),
    'multiclass': 150 * np.array([0, 1, 2, 3, 4]),
    'multilabel_native': 210 * np.array([0, 1, 2, 3, 4]),
    'multilabel_powerset': 540 * np.array([0, 1, 2, 3, 4])
}

for scenario, dataset_refs in datasets.items():
    df, data = pd.read_excel(f'{base_folder}/results/{scenario}/{scenario}.xlsx', sheet_name='training_time'), {}
    for dataset_ref in dataset_refs:
        data[dataset_ref] = df[dataset_ref].fillna('00:00:00 (00:00:00 ± 00:00:00)')

    pattern = r'(\d+:\d+:\d+)\s+\((\d+:\d+:\d+)\s*±\s*(\d+:\d+:\d+)\)'

    # Extract min, mean, and standard deviation values with error handling
    min_vals, mean_vals, stdev_vals = [], [], []
    for key in data.keys():
        min_row, mean_row, stdev_row = [], [], []
        for value in data[key]:
            match = re.match(pattern, value)
            if match:
                min_row.append(timestamp_to_seconds(match.group(1)))  # Min value in seconds
                mean_row.append(timestamp_to_seconds(match.group(2))) # Mean value in seconds
                stdev_row.append(timestamp_to_seconds(match.group(3))) # Stdev value in seconds
            else:
                # Append a default value (e.g., 0 seconds) if no match is found
                min_row.append(0)
                mean_row.append(0)
                stdev_row.append(0)
        min_vals.append(min_row)
        mean_vals.append(mean_row)
        stdev_vals.append(stdev_row)

    # Convert lists to arrays and set as integers
    min_vals = np.array(min_vals, dtype=int)
    mean_vals = np.array(mean_vals, dtype=int)
    stdev_vals = np.array(stdev_vals, dtype=int)

    print("Min values:", min_vals)
    print("Mean values:", mean_vals)
    print("Stdev values:", stdev_vals)

    # # Determine upper limit and y-ticks for the plot
    upper_limit = np.ceil(np.max([mean_vals + stdev_vals]) / 150) * 150
    upper_limit = np.ceil(upper_limit / 150) * 150  # Round up to the nearest multiple of 300
    scenario_y_ticks = y_ticks[scenario] # np.linspace(0, upper_limit, 6)
    print('upper_limit = ', upper_limit, 'y_ticks = ', scenario_y_ticks)

    # Set up positions for each bar
    positions = np.arange(len(data.keys())) / 1.25

    # Plotting
    fig, ax = plt.subplots(figsize=(22, 7))
    ax.yaxis.grid(True, linestyle='dashed', linewidth=0.5, alpha=0.7)

    # Fixed line at 300 seconds
    plt.axhline(y=300, color='red', linestyle='dashdot', linewidth=0.5, alpha=0.7)

    for i, framework in enumerate(frameworks):
        x_vals = positions + i * 0.04
        ax.bar(x_vals, mean_vals[:, i], width=0.04, label=framework, color=PLOT_COLORS[i])
        ax.errorbar(x_vals, mean_vals[:, i], yerr=stdev_vals[:, i], fmt='none', capsize=5, color='dimgray')
        
        # Check if min_vals is non-zero before plotting the marker
        non_zero_min_vals = [val if val != 0 else np.nan for val in min_vals[:, i]]
        ax.scatter(x_vals, non_zero_min_vals, marker='*', color='red')

    ax.set_xticks(positions + 0.04 * (len(frameworks) - 1) / 2)
    ax.set_xticklabels([str(x).replace('ps', '') for x in data.keys()], fontsize=20)
    ax.set_ylim(0, np.max(scenario_y_ticks) + scenario_y_ticks[1] / 2)
    ax.set_yticks(scenario_y_ticks)
    ax.set_yticklabels([seconds_to_timestamp(y) for y in scenario_y_ticks], fontsize=20)
    ax.set_xlabel(r'Dataset', fontsize=24)
    ax.set_ylabel(r'Training Time', fontsize=24)
    ax.legend(loc='best', bbox_to_anchor=(1, 1), fontsize=17, title='Frameworks', title_fontsize=18)

    plt.tight_layout()
    plt.savefig(f'{base_folder}/results/{scenario}/training_time_{scenario}.png', dpi=300)
