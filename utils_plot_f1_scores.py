import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import re
import warnings
from matplotlib import font_manager

warnings.filterwarnings("ignore", message=".*Matplotlib is currently using agg.*")
warnings.filterwarnings("ignore", message=".*FixedFormatter.*")

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

with open(f"{base_folder}/results/plot_colors.pkl", "rb") as f:
    PLOT_COLORS = pickle.load(f)

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

for scenario, dataset_refs in datasets.items():
    df, data = pd.read_excel(f'{base_folder}/results/{scenario}/{scenario}.xlsx', sheet_name='f1_score'), {}
    for dataset_ref in dataset_refs:
        data[dataset_ref] = df[dataset_ref].fillna('0.0 (0.0 ± 0.0)')

    pattern = r'(\d+\.\d+)\s+\((\d+\.\d+)\s*±\s*(\d+\.\d+)\)'

    # Extract max, mean, and standard deviation values with error handling
    max_vals, mean_vals, stdev_vals = [], [], []
    for key in data.keys():
        max_row, mean_row, stdev_row = [], [], []
        for value in data[key]:
            match = re.match(pattern, value)
            if match:
                max_row.append(float(match.group(1)))  # Max value
                mean_row.append(float(match.group(2))) # Mean value
                stdev_row.append(float(match.group(3))) # Stdev value
            else:
                # If no match is found, append default values (e.g., 0.0)
                max_row.append(0.0)
                mean_row.append(0.0)
                stdev_row.append(0.0)
        max_vals.append(max_row)
        mean_vals.append(mean_row)
        stdev_vals.append(stdev_row)

    # Convert lists to numpy arrays
    max_vals = np.array(max_vals)
    mean_vals = np.array(mean_vals)
    stdev_vals = np.array(stdev_vals)

    # Set up positions for each bar
    positions = np.arange(len(data.keys())) / 1.25

    # Plotting
    fig, ax = plt.subplots(figsize=(22, 7))
    ax.yaxis.grid(True, linestyle='dashed', linewidth=0.5, alpha=0.7)

    for i, framework in enumerate(frameworks):
        x_vals = positions + i * 0.04
        ax.bar(x_vals, mean_vals[:, i], width=0.04, label=framework, color=PLOT_COLORS[i])
        ax.errorbar(x_vals, mean_vals[:, i], yerr=stdev_vals[:, i], fmt='none', capsize=5, color='dimgray')
        
        # Check if max_vals is non-zero before plotting the marker
        non_zero_max_vals = [val if val != 0 else np.nan for val in max_vals[:, i]]
        ax.scatter(x_vals, non_zero_max_vals, marker='*', color='red')

    ax.set_xticks(positions + 0.04 * (len(frameworks) - 1) / 2)
    ax.set_xticklabels([str(x).replace('ps', '') for x in data.keys()], fontsize=20)
    ax.set_ylim(0., 1.1)
    ax.set_yticklabels([0., 0.2, 0.4, 0.6, 0.8, 1.], fontsize=20)
    ax.set_xlabel(r'Dataset', fontsize=24)
    ax.set_ylabel(r'Weighted $F_{1}$ Score', fontsize=24)
    ax.legend(loc='best', bbox_to_anchor=(1, 1), fontsize=17, title='Frameworks', title_fontsize=18)

    plt.tight_layout()
    plt.savefig(f'{base_folder}/results/{scenario}/f1_score_{scenario}.png', dpi=300)
