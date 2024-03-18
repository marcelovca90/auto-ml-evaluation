import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from matplotlib import font_manager
from matplotlib.cm import get_cmap

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

frameworks = ['4intelligence', 'AutoGluon', 'AutoKeras', 'Auto-PyTorch', 'AutoSklearn', 'EvalML', 'FLAML', 'GAMA', 'H2O', 'LightAutoML', 'PyCaret', 'TPOT']

for scenario, dataset_refs in datasets.items():

    df, data = pd.read_excel(f'{base_folder}/results/{scenario}.xlsx', sheet_name='f1_score'), {}
    for dataset_ref in dataset_refs:
        data[dataset_ref] = df[dataset_ref].fillna('0.0 (0.0 Â± 0.0)')

    pattern = r'(\d+\.\d+)\s+\((\d+\.\d+)\s*Â±\s*(\d+\.\d+)\)'

    # Extract max, mean, and standard deviation values
    max_vals = np.array([[float(re.match(pattern, value).group(1)) for value in data[key]] for key in data.keys()])
    mean_vals = np.array([[float(re.match(pattern, value).group(2)) for value in data[key]] for key in data.keys()])
    stdev_vals = np.array([[float(re.match(pattern, value).group(3)) for value in data[key]] for key in data.keys()])

    # Set up positions for each bar
    positions = np.arange(len(data.keys())) / 1.25

    # Get colormap
    cmap = get_cmap('Paired')

    # Plotting
    fig, ax = plt.subplots(figsize=(22, 7))
    ax.yaxis.grid(True, linestyle='dashed', linewidth=0.5, alpha=0.7)  # Enable y-grid with customized properties

    for i, framework in enumerate(frameworks):
        x_vals = positions + i * 0.05  # Adjust the width between bars
        color = cmap(i / len(frameworks))  # Color based on framework index
        ax.bar(x_vals, mean_vals[:, i], width=0.05, label=framework, color=color)
        ax.errorbar(x_vals, mean_vals[:, i], yerr=stdev_vals[:, i], fmt='none', capsize=5, color='dimgray')
        # Check if max_vals is non-zero before plotting the marker
        non_zero_max_vals = [val if val != 0 else np.nan for val in max_vals[:, i]]
        ax.scatter(x_vals, non_zero_max_vals, marker='*', color='red')

    ax.set_xticks(positions + 0.05 * (len(frameworks) - 1) / 2)  # Set ticks at the center of each group
    ax.set_xticklabels([str(x).replace('ps', '') for x in data.keys()], fontsize=20)  # Increase font size for x-axis labels
    ax.set_ylim(0., 1.1)
    ax.set_yticklabels([0., 0.2, 0.4, 0.6, 0.8, 1.], fontsize=20)  # Increase font size for x-axis labels
    ax.set_xlabel(r'Dataset', fontsize=24)  # Increase font size for x-axis label
    ax.set_ylabel(r'$F_{1}$ Score', fontsize=24)  # Increase font size for y-axis label
    ax.legend(loc='best', bbox_to_anchor=(1, 1), fontsize=20)  # Increase font size for legend
    # ax.set_title(r'$F_{1}$ Score (Max, Mean, Stdev) for each Framework and Dataset', fontsize=24)  # Increase font size for title

    plt.tight_layout()
    plt.savefig(f'{base_folder}/results/f1_score_{scenario}.png')
    plt.show()
