import collections
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
from common import *

metric = 'f1_score_weighted'

plt.figure(figsize=(32,24))
plt.rcParams['font.size'] = '16'

result_dict = {}

datasets = {
    'binary': [37, 44, 1462, 1479, 1510],
    'multiclass': [23, 181, 1466, 40691, 40975],
    # 'multilabel': [41465, 41468, 41470, 41471, 41473]
}

def format_time(secs):
    return time.strftime("%H:%M:%S", time.gmtime(secs))

# generate statistics xlsx for individual datasets
for scenario, dataset_refs in datasets.items():
    for dataset_ref in dataset_refs:
        # read jsons and aggregate results
        for filename in os.listdir(f"{base_folder}/results/{dataset_ref}"):
            if filename.endswith('.json'):
                with open(f'{base_folder}/results/{dataset_ref}/{filename}') as file:
                    content = json.load(file)

                    framework = content['framework']
                    
                    result_dict[framework] = {}

                    f1_score_weighted_values = [r['f1_score_weighted'] for r in content['results']]
                    f1_score_weighted_min = np.min(f1_score_weighted_values)
                    f1_score_weighted_max = np.max(f1_score_weighted_values)
                    f1_score_weighted_mean = np.mean(f1_score_weighted_values)
                    f1_score_weighted_median = np.median(f1_score_weighted_values)
                    f1_score_weighted_std = np.std(f1_score_weighted_values)
                    result_dict[framework]['f1_score_weighted'] = \
                        f'{f1_score_weighted_max:.3f} ({f1_score_weighted_mean:.3f} ± {f1_score_weighted_std:.3f})'
                    
                    training_time_values = [r['training_time'] for r in content['results']]
                    training_time_min = format_time(np.min(training_time_values))
                    training_time_max = format_time(np.max(training_time_values))
                    training_time_mean = format_time(np.mean(training_time_values))
                    training_time_median = format_time(np.median(training_time_values))
                    training_time_std = format_time(np.std(training_time_values))
                    result_dict[framework]['training_time'] = \
                        f'{training_time_min} ({training_time_mean} ± {training_time_std})'

                    test_time_values = [r['test_time'] for r in content['results']]
                    test_time_min = format_time(np.min(test_time_values))
                    test_time_max = format_time(np.max(test_time_values))
                    test_time_mean = format_time(np.mean(test_time_values))
                    test_time_median = format_time(np.median(test_time_values))
                    test_time_std = format_time(np.std(test_time_values))
                    result_dict[framework]['test_time'] = \
                        f'{test_time_min} ({test_time_mean} ± {test_time_std})'

        print(result_dict)
        # write csv, xlsx, and markdown files
        result_df = pd.DataFrame.from_dict(result_dict, orient='index')
        result_df.sort_index(inplace=True)
        result_df.to_csv(f'{base_folder}/results/{dataset_ref}/automl.csv', index=False)
        result_df.to_excel(f'{base_folder}/results/{dataset_ref}/automl.xlsx', index='framework', index_label='framework')
        result_df.to_markdown(f'{base_folder}/results/{dataset_ref}/automl.md', index=False, tablefmt='pipe')

# merge statistics xlsx for all datasets
for scenario, dataset_refs in datasets.items():
    # f1 score
    f1_merged = pd.DataFrame()
    for dataset_ref in dataset_refs:
        f1_single = pd.read_excel(f'{base_folder}/results/{dataset_ref}/automl.xlsx')
        f1_single = f1_single[['framework', 'f1_score_weighted']]
        f1_single = f1_single.rename(columns={"f1_score_weighted": dataset_ref})
        f1_merged = f1_single if f1_merged.empty else \
            pd.merge(f1_merged, f1_single, on="framework", how="outer")
        
    # training time
    training_time_merged = pd.DataFrame()
    for dataset_ref in dataset_refs:
        training_time_single = pd.read_excel(f'{base_folder}/results/{dataset_ref}/automl.xlsx')
        training_time_single = training_time_single[['framework', 'training_time']]
        training_time_single = training_time_single.rename(columns={"training_time": dataset_ref})
        training_time_merged = training_time_single if training_time_merged.empty else \
            pd.merge(training_time_merged, training_time_single, on="framework", how="outer")
    
    # test time
    test_time_merged = pd.DataFrame()
    for dataset_ref in dataset_refs:
        test_time_single = pd.read_excel(f'{base_folder}/results/{dataset_ref}/automl.xlsx')
        test_time_single = test_time_single[['framework', 'test_time']]
        test_time_single = test_time_single.rename(columns={"test_time": dataset_ref})
        test_time_merged = test_time_single if test_time_merged.empty else \
            pd.merge(test_time_merged, test_time_single, on="framework", how="outer")

    # consolidated results
    with pd.ExcelWriter(f'{base_folder}/results/{scenario}.xlsx', engine='openpyxl') as writer:
        f1_merged.round(3).to_excel(writer, sheet_name='f1_score', index=False)
        training_time_merged.to_excel(writer, sheet_name='training_time', index=False)
        test_time_merged.to_excel(writer, sheet_name='test_time', index=False)