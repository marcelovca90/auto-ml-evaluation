import json
import os
import time

import pandas as pd
from pytictoc import TicToc
from sklearn.datasets import fetch_openml
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

# for reference:
# - binary = 37 (diabetes) or 44 (spambase)
# - multiclass = 61 (iris) or 32 (pendigits)
DATASET_REF = 37
EXEC_TIME_MINUTES = 1
EXEC_TIME_SECONDS = EXEC_TIME_MINUTES*60
SEED = 42
TIMER = TicToc()

def infer_task_type(y_test):
    num_classes = len(set(y_test))
    if num_classes == 1:
        raise Exception('Malformed data set; num_classes = 1')
    elif num_classes == 2:
        task_type = 'binary'
    else:
        task_type = 'multiclass'
    return task_type

def load_data_delegate():
    if isinstance(DATASET_REF, int):
        return load_openml()
    elif isinstance(DATASET_REF, str):
        return load_csv()
    else:
        raise Exception('DATASET_REFERENCE must be int (OpenML) or str (local CSV)')

def load_csv(dataset_folder=DATASET_REF):
    base_folder = os.path.join(os.path.dirname(__file__), dataset_folder)
    filenames = ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']
    dfs = []
    for filename in filenames:
        full_filename = os.path.join(base_folder, filename)
        csv = pd.read_csv(filepath_or_buffer=full_filename).infer_objects().to_numpy()
        dfs.append(csv.ravel() if csv.shape[1] == 1 else csv)
    return dfs[0], dfs[1], dfs[2], dfs[3]

def load_openml(dataset_id=DATASET_REF):
    dataset = fetch_openml(data_id=dataset_id, return_X_y=False)
    X, y = dataset.data, pd.Series(pd.factorize(dataset.target)[0])
    return train_test_split(X, y, test_size=0.2, random_state=SEED)

def calculate_score(metric, y_true, y_pred, **kwargs):
	try:
		return metric(y_true, y_pred, **kwargs)
	except:
		return -1.0

def collect_and_persist_results(y_test, y_pred, training_time, test_time, framework="unknown"):
  results = {}
  results.update({"accuracy_score":          calculate_score(accuracy_score, y_test, y_pred)})
  results.update({"average_precision_score": calculate_score(average_precision_score, y_test, y_pred)})
  results.update({"balanced_accuracy_score": calculate_score(balanced_accuracy_score, y_test, y_pred)})
  results.update({"cohen_kappa_score":       calculate_score(cohen_kappa_score, y_test, y_pred)})
  results.update({"f1_score_macro":          calculate_score(f1_score, y_test, y_pred, average="macro")})
  results.update({"f1_score_micro":          calculate_score(f1_score, y_test, y_pred, average="micro")})
  results.update({"f1_score_weighted":       calculate_score(f1_score, y_test, y_pred, average="weighted")})
  results.update({"matthews_corrcoef":       calculate_score(matthews_corrcoef, y_test, y_pred)})
  results.update({"precision_score":         calculate_score(precision_score, y_test, y_pred)})
  results.update({"recall_score":            calculate_score(recall_score, y_test, y_pred)})
  results.update({"roc_auc_score":           calculate_score(roc_auc_score, y_test, y_pred)})
  results.update({"training_time": time.strftime("%H:%M:%S", time.gmtime(training_time))})
  results.update({"test_time": time.strftime("%H:%M:%S", time.gmtime(test_time))})
  print(results)
  with open(f"./results/automl_{framework}.json", "w") as outfile:
    json.dump(results, outfile)
