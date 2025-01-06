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
from skmultilearn.problem_transform import LabelPowerset

# pd.options.mode.chained_assignment = None

FORCE_DATASET_REF = None
EXEC_TIME_MINUTES = 5
EXEC_TIME_SECONDS = EXEC_TIME_MINUTES*60
MAX_MEMORY_MB = 32*1024
NUM_CPUS = multiprocessing.cpu_count()
PRIME_NUMBERS = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71
]
LABEL_POWERSET = True # use True when needed
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
    if FORCE_DATASET_REF:
        return FORCE_DATASET_REF
    elif len(sys.argv) < 2:
        print('usage: python common.py dataset_ref')
    else:
        dataset_ref = None
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
    return get_dataset_ref() in [285, 41464, 41465, 41468, 41470, 41471, 41473]

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
    dataset = fetch_openml(data_id=get_dataset_ref(), return_X_y=False, parser='auto')
    X, y = dataset.data.copy(deep=True), dataset.target.copy(deep=True)

    # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9321731
    # https://github.com/mrapp-ke/Boomer-Datasets/raw/refs/heads/main/flags.arff
    if get_dataset_ref() == 285: # flags
        df = pd.concat([X, y], axis='columns')
        label_columns = [
            'crescent', 'triangle', 'icon', 'animate', 'text', 'red',
            'green', 'blue', 'gold', 'white', 'black', 'orange'
        ]
        y = df[label_columns].astype(int)  # Select only label columns
        for col in y.columns.values:
            y[col] = y[col].map({0: 'FALSE', 1: 'TRUE'})
        X = df.drop(columns=label_columns).infer_objects()  # Drop label columns to get remaining ones
        for col in X.columns:
            if col not in ['mainhue', 'topleft', 'botright']:
                X[col] = X[col].astype(float)
        assert df.shape[0] == X.shape[0] # rows
        assert df.shape[0] == y.shape[0] # rows
        assert df.shape[1] == X.shape[1] + y.shape[1] # columns

    # handle categorical features
    for col in X.columns.values:
        if X[col].dtype.name == 'category':
            X.loc[:, col] = pd.Series(pd.factorize(X[col])[0])

    # handle target(s) column(s)
    if is_multi_label():
        for col in y.columns.values:
            y[col] = y[col].map({'FALSE': 0, 'TRUE': 1}).to_numpy()
        if LABEL_POWERSET:
            y = pd.Series(LabelPowerset().transform(y), name="class")
    else:
        y = pd.Series(pd.factorize(y)[0], name="class")

    return train_test_split(X, y, test_size=0.2, random_state=seed)

def calculate_score(metric, y_true, y_pred, **kwargs):
	try:
		return metric(y_true, y_pred, **kwargs)
	except:
		return -1.0

def collect_and_persist_results(y_test, y_pred, training_time, test_time, framework="unknown", seed=None):
    suffix = 'ps' if LABEL_POWERSET else ''
    results_folder = f'./results/{get_dataset_ref()}{suffix}'
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
    print(f"\n\n{'*' * 80}\n{framework} => {this_result}\n{'*' * 80}\n\n")
    all_results['results'].append(this_result)
    with open(results_filename, 'w') as outfile:
        json.dump(all_results, outfile)
