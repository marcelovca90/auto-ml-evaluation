import json
import os
import sys
import time

import pandas as pd
from pytictoc import TicToc
from sklearn.datasets import fetch_openml
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

EXEC_TIME_MINUTES = 10
EXEC_TIME_SECONDS = EXEC_TIME_MINUTES*60
SEED = 42
TIMER = TicToc()

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

def infer_task_type(y_test=None):
    if get_dataset_ref() in [41465, 41468, 41470, 41471, 41473]:
        task_type = 'multilabel'
    else:
        num_classes = len(set(y_test))
        if num_classes == 1:
            raise Exception('Malformed data set; num_classes == 1')
        elif num_classes == 2:
            task_type = 'binary'
        else:
            task_type = 'multiclass'
    return task_type

def load_data_delegate():
    if isinstance(get_dataset_ref(), int):
        return load_openml()
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

def load_openml():
    dataset = fetch_openml(data_id=get_dataset_ref(), return_X_y=False)
    if infer_task_type() == 'multilabel':
        X, y = dataset.data, dataset.target
        for col in y.columns.values:
            y[col] = y[col].map({'FALSE': 0, 'TRUE': 1}).to_numpy()
    else:
        X, y = dataset.data, pd.Series(pd.factorize(dataset.target)[0])
    return train_test_split(X, y, test_size=0.2, random_state=SEED)

def calculate_score(metric, y_true, y_pred, **kwargs):
	try:
		return metric(y_true, y_pred, **kwargs)
	except:
		return -1.0

def collect_and_persist_results(y_test, y_pred, training_time, test_time, framework="unknown"):
    results = {}
    results.update({"framework":                                framework})
    results.update({"accuracy_score":                           calculate_score(accuracy_score, y_test, y_pred)})
    results.update({"average_precision_score":                  calculate_score(average_precision_score, y_test, y_pred)})
    results.update({"balanced_accuracy_score":                  calculate_score(balanced_accuracy_score, y_test, y_pred)})
    results.update({"cohen_kappa_score":                        calculate_score(cohen_kappa_score, y_test, y_pred)})
    results.update({"f1_score_macro":                           calculate_score(f1_score, y_test, y_pred, average="macro")})
    results.update({"f1_score_micro":                           calculate_score(f1_score, y_test, y_pred, average="micro")})
    results.update({"f1_score_weighted":                        calculate_score(f1_score, y_test, y_pred, average="weighted")})
    results.update({"matthews_corrcoef":                        calculate_score(matthews_corrcoef, y_test, y_pred)})
    results.update({"precision_score":                          calculate_score(precision_score, y_test, y_pred)})
    results.update({"recall_score":                             calculate_score(recall_score, y_test, y_pred)})
    results.update({"roc_auc_score":                            calculate_score(roc_auc_score, y_test, y_pred)})
    results.update({"coverage_error":                           calculate_score(coverage_error, y_test, y_pred)})
    results.update({"label_ranking_average_precision_score":    calculate_score(label_ranking_average_precision_score, y_test, y_pred)})
    results.update({"label_ranking_loss":                       calculate_score(label_ranking_loss, y_test, y_pred)})
    results.update({"training_time":                            time.strftime("%H:%M:%S", time.gmtime(training_time))})
    results.update({"test_time":                                time.strftime("%H:%M:%S", time.gmtime(test_time))})
    print(results)
    if not os.path.exists(f'./results/{get_dataset_ref()}'):
        os.makedirs(f'./results/{get_dataset_ref()}')
    with open(f"./results/{get_dataset_ref()}/automl_{framework}.json", "w") as outfile:
        json.dump(results, outfile)
