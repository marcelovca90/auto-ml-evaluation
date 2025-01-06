import json
import os

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