import json
import multiprocessing
import os
import sys
import time

import pandas as pd
from pytictoc import TicToc
from sklearn.datasets import fetch_openml
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

EXEC_TIME_MINUTES = 1
EXEC_TIME_SECONDS = EXEC_TIME_MINUTES*60
NUM_CPUS = multiprocessing.cpu_count()
PRIME_NUMBERS = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    # 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    # 127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    # 179, 181, 191, 193, 197, 199, 211, 223, 227, 229
]
TIMER = TicToc()

def set_random_seed(seed):
    try:
        import random
        random.seed(seed)
    except ImportError as e:
        print(f"Error importing a necessary module; skipping seed: {e}")
    try:
        import numpy
        numpy.random.seed(seed)
    except ImportError as e:
        print(f"Error importing a necessary module; skipping seed: {e}")
    try:
        from sklearn.utils import check_random_state
        check_random_state(seed)
    except ImportError as e:
        print(f"Error importing a necessary module; skipping seed: {e}")
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError as e:
        print(f"Error importing a necessary module; skipping seed: {e}")
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError as e:
        print(f"Error importing a necessary module; skipping seed: {e}")

def get_dataset_ref():
    dataset_ref = None
    if len(sys.argv) != 2:
        print('usage: python common.py dataset_ref')
    else:
        try:
            dataset_ref = int(sys.argv[1])
        except:
            dataset_ref = str(sys.argv[1])
    return dataset_ref

def infer_task_type(y_test):
    num_classes = len(set(y_test))
    if num_classes == 1:
        raise Exception('Malformed data set; num_classes == 1')
    elif num_classes == 2:
        task_type = 'binary'
    else:
        task_type = 'multiclass'
    return task_type

def is_multi_label():
    return get_dataset_ref() in [41465, 41468, 41470, 41471, 41473]

def load_data_delegate(seed):
    if isinstance(get_dataset_ref(), int):
        return load_openml(seed)
    elif isinstance(get_dataset_ref(), str):
        return load_csv()
    else:
        raise Exception('dataset_ref must be int (OpenML) or str (local CSV)')

def load_csv():
    base_folder = os.path.join(os.path.dirname(__file__), 'datasets', get_dataset_ref())
    filenames = ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']
    dfs = []
    for filename in filenames:
        full_filename = os.path.join(base_folder, filename)
        csv = pd.read_csv(filepath_or_buffer=full_filename).infer_objects().to_numpy()
        dfs.append(csv.ravel() if csv.shape[1] == 1 else csv)
    return dfs[0], dfs[1], dfs[2], dfs[3]

def load_openml(seed):
    dataset = fetch_openml(data_id=get_dataset_ref(), return_X_y=False)
    if is_multi_label():
        X, y = dataset.data, dataset.target
        for col in y.columns.values:
            y[col] = y[col].map({'FALSE': 0, 'TRUE': 1}).to_numpy()
    else:
        X, y = dataset.data, dataset.target
        for col in X.columns.values:
            if X[col].dtype.name == 'category':
                X[col] = pd.Series(pd.factorize(X[col])[0])
        y = pd.Series(pd.factorize(y)[0])
    return train_test_split(X, y, test_size=0.2, random_state=seed)

def calculate_score(metric, y_true, y_pred, **kwargs):
	try:
		return metric(y_true, y_pred, **kwargs)
	except:
		return -1.0

def collect_and_persist_results(y_test, y_pred, training_time, test_time, framework="unknown", seed=None):
    results_folder = f'./results/{get_dataset_ref()}'
    results_filename = f'{results_folder}/automl_{framework}.json'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    if os.path.exists(results_filename):
        with open(results_filename, 'r') as infile:
            all_results = json.load(infile)
    else:
        all_results = {
            'id': get_dataset_ref(),
            'framework': framework,
            'results': []
        }
    this_result = {
        "seed":                                     str(seed),
        "accuracy_score":                           calculate_score(accuracy_score, y_test, y_pred),
        "average_precision_score":                  calculate_score(average_precision_score, y_test, y_pred),
        "balanced_accuracy_score":                  calculate_score(balanced_accuracy_score, y_test, y_pred),
        "cohen_kappa_score":                        calculate_score(cohen_kappa_score, y_test, y_pred),
        "f1_score_macro":                           calculate_score(f1_score, y_test, y_pred, average="macro"),
        "f1_score_micro":                           calculate_score(f1_score, y_test, y_pred, average="micro"),
        "f1_score_weighted":                        calculate_score(f1_score, y_test, y_pred, average="weighted"),
        "matthews_corrcoef":                        calculate_score(matthews_corrcoef, y_test, y_pred),
        "precision_score":                          calculate_score(precision_score, y_test, y_pred),
        "recall_score":                             calculate_score(recall_score, y_test, y_pred),
        "roc_auc_score":                            calculate_score(roc_auc_score, y_test, y_pred),
        "coverage_error":                           calculate_score(coverage_error, y_test, y_pred),
        "label_ranking_average_precision_score":    calculate_score(label_ranking_average_precision_score, y_test, y_pred),
        "label_ranking_loss":                       calculate_score(label_ranking_loss, y_test, y_pred),
        "training_time":                            training_time,
        "test_time":                                test_time
    }
    print(this_result)
    all_results['results'].append(this_result)
    with open(results_filename, 'w') as outfile:
        json.dump(all_results, outfile)
