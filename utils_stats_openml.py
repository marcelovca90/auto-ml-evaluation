import json
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.datasets import fetch_openml
from skmultilearn.problem_transform import LabelPowerset
from ydata_profiling import ProfileReport

dataset_ids = {
    'binary': [31, 37, 44, 1462, 1479, 1510, 40945],
    'multiclass': [23, 36, 54, 181, 1466, 40691, 40975],
    'multilabel': [285, 41464, 41465, 41468, 41470, 41471, 41473]
}

def get_data_type(dataset_type, dataset_id, X):
    profile = ProfileReport(X)
    profile.to_file(f"reports/{dataset_type}_{dataset_id}.html")
    data_types = {k:v['type'] for k,v in profile.description_set.variables.items()}
    numerical_cols = [col for col, dtype in data_types.items() if dtype == "Numeric"]
    categorical_cols = [col for col, dtype in data_types.items() if dtype == "Categorical"]
    data_type = "Unknown"
    if len(numerical_cols) == len(data_types):  # All columns are numerical (quantitative)
        data_type = "Quantitative"
    elif len(categorical_cols) == len(data_types):  # All columns are categorical (qualitative)
        data_type = "Qualitative"
    else:  # Mixed
        data_type = "Mixed"
    return data_type

def calculate_complexity_binary_multiclass(n_samples, n_features, n_classes):
    complexity = (n_features * n_classes) / n_samples
    return round(complexity, 3)

def calculate_complexity_multilabel(n_samples, n_features, y):
    total_labels = y.columns.size
    total_labels_assigned = y.values.sum()
    num_samples = len(y)
    label_cardinality = total_labels_assigned / num_samples
    effective_classes = label_cardinality * total_labels
    complexity = (n_features * effective_classes) / n_samples
    return round(complexity, 3)

def load_openml(dataset_type, dataset_id):
    dataset = fetch_openml(data_id=dataset_id, return_X_y=False)
    X, y = dataset.data.copy(deep=True), dataset.target.copy(deep=True)

    # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9321731
    # https://github.com/mrapp-ke/Boomer-Datasets/raw/refs/heads/main/flags.arff
    if dataset_id == 285: # flags
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

    id, name = dataset.details['id'], dataset.details['name']
    data_type = get_data_type(dataset_type, dataset_id, X)
    n_samples, n_features = X.shape[0], X.shape[1]
    clf_type, n_classes, complexity = None, np.nan, np.nan
    if isinstance(y, pd.Series): # binary or multiclass
        n_classes = y.nunique()
        clf_type = 'binary' if n_classes == 2 else 'multiclass'
        complexity = calculate_complexity_binary_multiclass(n_samples, n_features, n_classes)
    elif isinstance(y, pd.DataFrame): # multilabel
        n_classes = y.columns.size
        clf_type = 'multilabel'
        for col in y.columns.values:
            y[col] = y[col].map({'FALSE': 0, 'TRUE': 1}).to_numpy()
        complexity = calculate_complexity_multilabel(n_samples, n_features, y)

    latex_str = ' & '.join([str(x) for x in [id, name, data_type, n_samples, n_features, n_classes, complexity]])

    result = {
        "type": clf_type,
        "id": id,
        "name": name,
        "data_type": data_type,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "complexity": complexity,
        "latex_str": latex_str
    }
    pprint(result, width=200)
    return result

def calculate_powerset(dataset_id):
    dataset = fetch_openml(data_id=dataset_id, return_X_y=False, parser='auto')
    X, y = dataset.data.copy(deep=True), dataset.target.copy(deep=True)

    # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9321731
    # https://github.com/mrapp-ke/Boomer-Datasets/raw/refs/heads/main/flags.arff
    if dataset_id == 285: # flags
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

    for col in y.columns.values:
        y[col] = y[col].map({'FALSE': 0, 'TRUE': 1}).to_numpy()
        
    y_ps = pd.Series(LabelPowerset().transform(y))
    print(dataset_id, y_ps.nunique(), 'unique labels')

def merge_jsons():
    summary = {}
    for ds_type in dataset_ids.keys():
        with open(f'reports/{ds_type}.json', 'r', encoding='utf-8') as fp:
            content = json.load(fp)
        summary[ds_type] = content

    with open('reports/summary.json', 'w', encoding='utf-8') as fp:
        json.dump(summary, fp, indent=4, ensure_ascii=True)


if __name__ == "__main__":

    # summary = {}
    # for ds_type, ds_ids in dataset_ids.items():
    #     summary[ds_type] = []
    #     print(ds_type)
    #     for id in ds_ids:
    #         result = load_openml(ds_type, id)
    #         summary[ds_type].append(result)
    #     with open(f'reports/{ds_type}.json', 'w', encoding='utf-8') as fp:
    #         json.dump(summary[ds_type], fp, indent=4, ensure_ascii=True)
    # with open('reports/summary.json', 'w', encoding='utf-8') as fp:
    #     json.dump(summary, fp, indent=4, ensure_ascii=True)

    merge_jsons()
