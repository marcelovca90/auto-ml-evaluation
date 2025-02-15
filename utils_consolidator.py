import distinctipy
import json
import numpy as np
import os
import pandas as pd
import shutil
import time
from common import *
import matplotlib.pyplot as plt

metric = 'f1_score_weighted'

base_folder = './' # '//wsl.localhost/Ubuntu/home/marce/git/auto-ml-comparison-2024/'

datasets = {
    'binary': [31, 37, 44, 1462, 1479, 1510, 40945],
    'multiclass': [23, 36, 54, 181, 1466, 40691, 40975],
    'multilabel_native': [285, 41464, 41465, 41468, 41470, 41471, 41473],
    'multilabel_powerset': ['285ps', '41464ps', '41465ps', '41468ps', '41470ps', '41471ps', '41473ps']
}

frameworks = [
    '4intelligence', 'autogluon', 'autokeras', 'autopytorch', 'autosklearn', 'evalml', 'fedot',
    'flaml', 'gama', 'h2o', 'lightautoml', 'lightwood', 'mljar', 'naive', 'pycaret', 'tpot',
]

PLOT_COLORS = distinctipy.get_colors(
    n_colors=len(frameworks), n_attempts=1_000, colorblind_type="Deuteranomaly", exclude_colors=[(0,0,0), (1,1,1), (1,0,0)], rng=42
)

def format_time(secs):
    return time.strftime("%H:%M:%S", time.gmtime(secs))

def get_best_seed(content):
    id = content['id']
    best_seed, best_f1_score = -1, -1
    for result in content['results']:
        current_seed = result['seed']
        current_f1_score = result['f1_score_weighted']
        print(f"[{id}] seed = {current_seed} => f1_score_weighted = {current_f1_score}")
        if current_f1_score > best_f1_score:
            best_seed, best_f1_score = current_seed, current_f1_score
            print(f"[{id}] best_seed_so_far = {best_seed} => best_f1_score_weighted_so_far = {best_f1_score}")
    print(f"[{id}] best_seed = {best_seed} => best_f1_score_weighted = {best_f1_score}")
    return best_seed

################################################################################

try:

    print("Step 1: consolidate results...", end="")

    result_dict = {}

    for scenario, dataset_refs in datasets.items():
        result_dict[scenario] = {}
        plt.figure(figsize=(12, 8))
        for dataset_ref in dataset_refs:
            result_dict[scenario][dataset_ref] = {}
            for framework in frameworks:
                result_dict[scenario][dataset_ref][framework] = {}
                content_json_filename = f'{base_folder}/results/{dataset_ref}/automl_{framework}.json'
                if os.path.exists(content_json_filename):
                    with open(content_json_filename) as file:
                        content = json.load(file)

                        # Retrieve f1_score_weighted values
                        f1_score_weighted_values = [r['f1_score_weighted'] for r in content['results']]

                        # Check if all f1_score_weighted values are -1
                        if all(score == -1 for score in f1_score_weighted_values):
                            print(f"Discarding {content_json_filename} as all f1_score_weighted values are -1.")
                            result_dict[scenario][dataset_ref][framework]['f1_score_weighted'] = None
                            result_dict[scenario][dataset_ref][framework]['training_time'] = None
                            result_dict[scenario][dataset_ref][framework]['test_time'] = None
                            result_dict[scenario][dataset_ref][framework]['missing_runs'] = None
                            result_dict[scenario][dataset_ref][framework]['best_seed'] = None
                            continue  # Skip further processing for this file
                        
                        f1_score_weighted_values = [r['f1_score_weighted'] for r in content['results']]
                        f1_score_weighted_min = np.min(f1_score_weighted_values)
                        f1_score_weighted_max = np.max(f1_score_weighted_values)
                        f1_score_weighted_mean = np.mean(f1_score_weighted_values)
                        f1_score_weighted_median = np.median(f1_score_weighted_values)
                        f1_score_weighted_std = np.std(f1_score_weighted_values)
                        result_dict[scenario][dataset_ref][framework]['f1_score_weighted'] = \
                            f'{f1_score_weighted_max:.3f} ({f1_score_weighted_mean:.3f} ± {f1_score_weighted_std:.3f})'

                        training_time_values = [r['training_time'] for r in content['results']]
                        training_time_min = format_time(np.min(training_time_values))
                        training_time_max = format_time(np.max(training_time_values))
                        training_time_mean = format_time(np.mean(training_time_values))
                        training_time_median = format_time(np.median(training_time_values))
                        training_time_std = format_time(np.std(training_time_values))
                        result_dict[scenario][dataset_ref][framework]['training_time'] = \
                            f'{training_time_min} ({training_time_mean} ± {training_time_std})'

                        test_time_values = [r['test_time'] for r in content['results']]
                        test_time_min = format_time(np.min(test_time_values))
                        test_time_max = format_time(np.max(test_time_values))
                        test_time_mean = format_time(np.mean(test_time_values))
                        test_time_median = format_time(np.median(test_time_values))
                        test_time_std = format_time(np.std(test_time_values))
                        result_dict[scenario][dataset_ref][framework]['test_time'] = \
                            f'{test_time_min} ({test_time_mean} ± {test_time_std})'
                        
                        result_dict[scenario][dataset_ref][framework]['missing_runs'] = \
                            [int(r['seed']) for r in content['results']]
                        result_dict[scenario][dataset_ref][framework]['missing_runs'] = \
                            [x for x in PRIME_NUMBERS if x not in result_dict[scenario][dataset_ref][framework]['missing_runs']]
                        
                        result_dict[scenario][dataset_ref][framework]['best_seed'] = get_best_seed(content)
                else:
                    result_dict[scenario][dataset_ref][framework]['f1_score_weighted'] = None
                    result_dict[scenario][dataset_ref][framework]['training_time'] = None
                    result_dict[scenario][dataset_ref][framework]['test_time'] = None
                    result_dict[scenario][dataset_ref][framework]['missing_runs'] = None
                    result_dict[scenario][dataset_ref][framework]['best_seed'] = None
            
            print(result_dict)
            result_df = pd.DataFrame.from_dict(result_dict[scenario][dataset_ref], orient='index')
            result_df.sort_index(inplace=True)
            result_df.to_excel(f'{base_folder}/results/{dataset_ref}/automl.xlsx', index='framework', index_label='framework')

        f1_merged = pd.DataFrame()
        training_time_merged = pd.DataFrame()
        test_time_merged = pd.DataFrame()
        missing_runs_merged = pd.DataFrame()
        best_seed_merged = pd.DataFrame()

        for dataset_ref in dataset_refs:

            if os.path.exists(f'{base_folder}/results/{dataset_ref}/automl.xlsx'):

                f1_single = pd.read_excel(f'{base_folder}/results/{dataset_ref}/automl.xlsx')
                f1_single = f1_single[['framework', 'f1_score_weighted']]
                f1_single = f1_single.rename(columns={"f1_score_weighted": dataset_ref})
                f1_merged = f1_single if f1_merged.empty else \
                    pd.merge(f1_merged, f1_single, on="framework", how="outer")

                training_time_single = pd.read_excel(f'{base_folder}/results/{dataset_ref}/automl.xlsx')
                training_time_single = training_time_single[['framework', 'training_time']]
                training_time_single = training_time_single.rename(columns={"training_time": dataset_ref})
                training_time_merged = training_time_single if training_time_merged.empty else \
                    pd.merge(training_time_merged, training_time_single, on="framework", how="outer")

                test_time_single = pd.read_excel(f'{base_folder}/results/{dataset_ref}/automl.xlsx')
                test_time_single = test_time_single[['framework', 'test_time']]
                test_time_single = test_time_single.rename(columns={"test_time": dataset_ref})
                test_time_merged = test_time_single if test_time_merged.empty else \
                    pd.merge(test_time_merged, test_time_single, on="framework", how="outer")

                missing_runs_single = pd.read_excel(f'{base_folder}/results/{dataset_ref}/automl.xlsx')
                missing_runs_single = missing_runs_single[['framework', 'missing_runs']]
                missing_runs_single = missing_runs_single.rename(columns={"missing_runs": dataset_ref})
                missing_runs_merged = missing_runs_single if missing_runs_merged.empty else \
                    pd.merge(missing_runs_merged, missing_runs_single, on="framework", how="outer")

                best_seed_single = pd.read_excel(f'{base_folder}/results/{dataset_ref}/automl.xlsx')
                best_seed_single = best_seed_single[['framework', 'best_seed']]
                best_seed_single = best_seed_single.rename(columns={"best_seed": dataset_ref})
                best_seed_merged = best_seed_single if best_seed_merged.empty else \
                    pd.merge(best_seed_merged, best_seed_single, on="framework", how="outer")

        with pd.ExcelWriter(f'{base_folder}/results/{scenario}.xlsx', engine='openpyxl') as writer:
            f1_merged.round(3).fillna('N/A').to_excel(writer, sheet_name='f1_score', index=False)
            training_time_merged.fillna('N/A').to_excel(writer, sheet_name='training_time', index=False)
            test_time_merged.fillna('N/A').to_excel(writer, sheet_name='test_time', index=False)
            missing_runs_merged.fillna('N/A').to_excel(writer, sheet_name='missing_runs', index=False)
            best_seed_merged.fillna('N/A').to_excel(writer, sheet_name='best_seed', index=False)

    print("OK")

except Exception as e:
    print("Error in Step 1:", str(e))

################################################################################

try: 
    print("Step 2: organize folders", end="")
    for scenario, dataset_refs in datasets.items():
        os.makedirs(f'{base_folder}/results/{scenario}', exist_ok=True)
        xlsx_src_path = f'{base_folder}/results/{scenario}.xlsx'
        xlsx_dst_path = f'{base_folder}/results/{scenario}/{scenario}.xlsx'
        shutil.move(xlsx_src_path, xlsx_dst_path)
        for dataset_ref in dataset_refs:
            folder_src_path = f'{base_folder}/results/{dataset_ref}'
            folder_dst_path = f'{base_folder}/results/{scenario}/{dataset_ref}'
            shutil.move(folder_src_path, folder_dst_path)
    print("OK")
except Exception as e:
    print("Error in Step 2:", str(e))

################################################################################

try: 
    print("Step 3: format JSONs", end="")
    for scenario, dataset_refs in datasets.items():
        for dataset_ref in dataset_refs:
            for framework in frameworks:
                content_json_filename = f'{base_folder}/results/{scenario}/{dataset_ref}/automl_{framework}.json'
                if os.path.exists(content_json_filename):
                    content = None
                    with open(content_json_filename, 'r') as file:
                        content = json.load(file)
                        print(f'Loaded {content_json_filename}.')
                    with open(content_json_filename, 'w', encoding='utf-8') as file:
                        json.dump(content, file, ensure_ascii=True, indent=4)
                        print(f'Formatted {content_json_filename}.')
    print("OK")
except Exception as e:
    print("Error in Step 3:", str(e))

################################################################################
