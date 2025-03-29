#!/bin/bash

# +---------------------------------------------------------------------------------------------------------------------------+
# |                                 OpenML Datasets (https://www.openml.org/search?type=data)                                 |
# +------------+--------+-----------------------------+----------------+-----------+----------+------------------+------------+
# | Type       | ID     | Dataset Name                | Data Type      | Instances | Features | Classes (Labels) | Complexity |
# |------------|--------|-----------------------------|----------------|-----------|----------|------------------|------------|
# |            | 31     | credit-g                    | Mixed          | 1,000     | 20       | 2                | 0.040      |
# |            | 37     | diabetes                    | Quantitative   | 768       | 8        | 2                | 0.021      |
# |            | 44     | spambase                    | Quantitative   | 4,601     | 57       | 2                | 0.025      |
# | Binary     | 1462   | bank-note-authentication    | Quantitative   | 1,372     | 4        | 2                | 0.006      |
# |            | 1479   | hill-valley                 | Quantitative   | 1,212     | 100      | 2                | 0.165      |
# |            | 1510   | wdbc                        | Quantitative   | 569       | 30       | 2                | 0.105      |
# |            | 40945  | titanic                     | Mixed          | 1,309     | 13       | 2                | 0.020      |
# |------------|--------|-----------------------------|----------------|-----------|----------|------------------|------------|
# |            | 23     | contraceptive-method-choice | Mixed          | 1,473     | 9        | 3                | 0.018      |
# |            | 36     | segment                     | Mixed          | 2,310     | 19       | 7                | 0.058      |
# |            | 54     | vehicle                     | Quantitative   | 846       | 18       | 4                | 0.085      |
# | Multiclass | 181    | yeast                       | Mixed          | 1,484     | 8        | 10               | 0.054      |
# |            | 1466   | cardiotocography            | Mixed          | 2,126     | 35       | 10               | 0.165      |
# |            | 40691  | wine-quality-red            | Quantitative   | 1,599     | 11       | 6                | 0.041      |
# |            | 40975  | car                         | Qualitative    | 1,728     | 6        | 4                | 0.014      |
# |------------|--------|-----------------------------|----------------|-----------|----------|------------------|------------|
# |            | 285    | flags                       | Mixed          | 194       | 17       | 12 (103)         | 4.336      |
# |            | 41464  | birds                       | Mixed          | 645       | 260      | 19 (133)         | 7.766      |
# |            | 41465  | emotions                    | Mixed          | 593       | 72       | 6 (27)           | 1.361      |
# | Multilabel | 41468  | image                       | Quantitative   | 2,000     | 135      | 5 (20)           | 0.417      |
# |            | 41470  | reuters                     | Mixed          | 2,000     | 243      | 7 (25)           | 0.981      |
# |            | 41471  | scene                       | Quantitative   | 2,407     | 294      | 6 (15)           | 0.787      |
# |            | 41473  | yeast                       | Quantitative   | 2,417     | 103      | 14 (198)         | 2.528      |
# +------------+--------+-----------------------------+----------------+-----------+----------+------------------+------------+

# datasets=(285)
datasets=(41464 41465 41468 41470 41471 41473)

mkdir /tmp/joblib &> /dev/null
NUM_CPUS=$(nproc)
export JOBLIB_TEMP_FOLDER=/tmp/joblib
export OPENBLAS_NUM_THREADS=$NUM_CPUS
export NUMEXPR_MAX_THREADS=$NUM_CPUS
export OMP_NUM_THREADS=$NUM_CPUS
export MKL_NUM_THREADS=$NUM_CPUS

echo Script execution started at $(date).

# Preparation
echo ======== Preparation ========
echo Started cleaning files from previous executions at $(date).
rm -rf __pycache* &> /dev/null
rm -rf Autogluon* &> /dev/null
rf -rf gama_* &> /dev/null
rm -rf mlruns* &> /dev/null
rm -rf structured* &> /dev/null
# rm -rf results/* &> /dev/null
# rm -rf venv-* &> /dev/null
rm *.log &> /dev/null
find . -maxdepth 1 -type d -name "*AutoML*" | xargs rm -rf
find . -maxdepth 1 -type d -name "*gama_*" | xargs rm -rf
echo Finished cleaning files from previous executions at $(date).

# Utils
echo ======== Utils ========
python3.8 -m venv venv-utils
source ./venv-utils/bin/activate
python3.8 -m pip install --upgrade pip
python3.8 -m pip install --upgrade setuptools pytictoc wheel pandas scikit-learn scikit-multilearn ydata-profiling distinctipy matplotlib tabulate openpyxl
python3.8 ./utils_consolidator.py
python3.8 ./utils_format_json.py
python3.8 ./utils_plot_f1_scores.py
python3.8 ./utils_plot_training_times.py

echo Script execution finished at $(date).
