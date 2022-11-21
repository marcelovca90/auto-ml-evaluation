from pytictoc import TicToc
from sklearn.datasets import fetch_openml
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import json
import os
import pandas as pd
import sklearn

SEED = 42

TIMER = TicToc()

def load_openml(dataset_id=44):

    X, y = sklearn.datasets.fetch_openml(data_id=dataset_id, return_X_y=True, as_frame=False)

    y = LabelEncoder().fit_transform(y)

    return train_test_split(X, y, test_size=0.2, random_state=SEED)

def calculate_score(metric, y_true, y_pred, **kwargs):
	try:
		return metric(y_true, y_pred, **kwargs)
	except:
		return -1.0

def collect_and_persist_scores(y_test, y_pred, key="unknown"):
  results = {}
  results.update({'accuracy_score': calculate_score(accuracy_score, y_test, y_pred)})
  results.update({'average_precision_score': calculate_score(average_precision_score, y_test, y_pred)})
  results.update({'balanced_accuracy_score': calculate_score(balanced_accuracy_score, y_test, y_pred)})
  results.update({'f1_score_micro': calculate_score(f1_score, y_test, y_pred, average='micro')})
  results.update({'f1_score_macro': calculate_score(f1_score, y_test, y_pred, average='macro')})
  results.update({'f1_score_weighted': calculate_score(f1_score, y_test, y_pred, average='weighted')})
  results.update({'precision_score': calculate_score(precision_score, y_test, y_pred)})
  results.update({'recall_score': calculate_score(recall_score, y_test, y_pred)})
  results.update({'roc_auc_score': calculate_score(roc_auc_score, y_test, y_pred)})
  print(results)
  with open(f"automl_{key}.json", "w") as outfile:
    json.dump(results, outfile)

def load_csv(dataset_folder, filename):
    full_filename = os.path.join(dataset_folder, filename)
    df = pd.read_csv(filepath_or_buffer=full_filename).infer_objects().to_numpy()
    return df.ravel() if df.shape[1] == 1 else df