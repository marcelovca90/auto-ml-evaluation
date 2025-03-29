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

datasets=(31 37 44 1462 1479 1510 40945 23 36 54 181 1466 40691 40975 285 41464 41465 41468 41470 41471 41473)

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
rm -rf results/* &> /dev/null
rm -rf venv-* &> /dev/null
rm *.log &> /dev/null
find . -maxdepth 1 -type d -name "*AutoML*" | xargs rm -rf
find . -maxdepth 1 -type d -name "*gama_*" | xargs rm -rf
echo Finished cleaning files from previous executions at $(date).

for id in ${datasets[@]}; do

    echo ======== Processing ========
    echo Started processing dataset $id at $(date).

    # AutoGluon
    echo ======== AutoGluon ========
    python3.8 -m venv venv-autogluon
    source ./venv-autogluon/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel pandas scikit-learn scikit-multilearn "spacy<3.8" "blis<1.0" torch==1.12+cpu torchvision==0.13.0+cpu torchtext==0.13.0 "autogluon<1.2" -f https://download.pytorch.org/whl/cpu/torch_stable.html
    python3.8 ./automl_autogluon.py $id
    pkill -f autogluon
    sleep 10

    # AutoKeras
    echo ======== AutoKeras ========
    python3.8 -m venv venv-autokeras
    source ./venv-autokeras/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel pandas scikit-learn scikit-multilearn git+https://github.com/keras-team/keras-tuner.git autokeras
    python3.8 ./automl_autokeras.py $id
    pkill -f autokeras
    sleep 10

    # AutoPyTorch
    echo ======== AutoPyTorch ========
    python3.8 -m venv venv-autopytorch
    source ./venv-autopytorch/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel pandas scikit-learn scikit-multilearn swig torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu "pynisher<1.0" "autoPyTorch<0.2"
    sed -i 's/applymap/astype/' ./venv-autopytorch/lib/python3.8/site-packages/autoPyTorch/data/tabular_feature_validator.py
    python3.8 ./automl_autopytorch.py $id
    pkill -f autopytorch
    sleep 10

    # AutoSklearn
    echo ======== AutoSklearn ========
    python3.8 -m venv venv-autosklearn
    source ./venv-autosklearn/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel pandas scikit-learn scikit-multilearn auto-sklearn
    python3.8 ./automl_autosklearn.py $id
    pkill -f autosklearn
    sleep 10

    # EvalML
    echo ======== EvalML ========
    python3.8 -m venv venv-evalml
    source ./venv-evalml/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel pandas scikit-learn scikit-multilearn evalml
    python3.8 ./automl_evalml.py $id
    pkill -f evalml
    sleep 10

    # FEDOT
    echo ======== FEDOT ========
    python3.8 -m venv venv-fedot
    source ./venv-fedot/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel pandas scikit-learn scikit-multilearn fedot fedot[extra]
    python3.8 ./automl_fedot.py $id
    pkill -f fedot
    sleep 10

    # FLAML
    echo ======== FLAML ========
    python3.8 -m venv venv-flaml
    source ./venv-flaml/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel pandas scikit-learn scikit-multilearn flaml[automl]
    python3.8 ./automl_flaml.py $id
    pkill -f flaml
    sleep 10

    # GAMA
    echo ======== GAMA ========
    python3.8 -m venv venv-gama
    source ./venv-gama/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel pandas scikit-learn scikit-multilearn gama
    sed -i 's/ SCORERS/ _SCORERS/' ./venv-gama/lib/python3.8/site-packages/gama/utilities/metrics.py
    python3.8 ./automl_gama.py $id
    pkill -f gama
    sleep 10

    # H2O
    echo ======== H2O ========
    python3.8 -m venv venv-h2o
    source ./venv-h2o/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel pandas scikit-learn scikit-multilearn requests tabulate future h2o
    python3.8 ./automl_h2o.py $id
    pkill -f h2o
    sleep 10

    # LightAutoML
    echo ======== LightAutoML ========
    python3.8 -m venv venv-lightautoml
    source ./venv-lightautoml/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel pandas scikit-learn scikit-multilearn lightautoml
    python3.8 ./automl_lightautoml.py $id
    pkill -f lightautoml
    sleep 10

    # Lightwood
    echo ======== Lightwood ========
    python3.8 -m venv venv-lightwood
    source ./venv-lightwood/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel pandas scikit-learn scikit-multilearn lightwood
    sed -i "s/sparse=False, //" venv-lightwood/lib/python3.8/site-packages/lightwood/encoder/helpers.py
    sed -i "s/sparse=False, //" venv-lightwood/lib/python3.8/site-packages/lightwood/analysis/nc/calibrate.py
    python3.8 ./automl_lightwood.py $id
    pkill -f lightwood
    sleep 10

    # MLJAR-supervised
    echo ======== MLJAR-supervised ========
    python3.8 -m venv venv-mljar
    source ./venv-mljar/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel pandas scikit-learn scikit-multilearn mljar-supervised
    python3.8 ./automl_mljar.py $id
    pkill -f mljar
    sleep 10

    # NaiveAutoML
    echo ======== NaiveAutoML ========
    python3.8 -m venv venv-naive
    source ./venv-naive/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel pandas scikit-learn scikit-multilearn naiveautoml
    python3.8 ./automl_naive.py $id
    pkill -f naive
    sleep 10

    # PyCaret
    echo ======== PyCaret ========
    python3.8 -m venv venv-pycaret
    source ./venv-pycaret/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel pandas scikit-learn scikit-multilearn "joblib<1.4" pycaret[full]
    python3.8 ./automl_pycaret.py $id
    pkill -f pycaret
    sleep 10

    # TPOT
    echo ======== TPOT ========
    python3.8 -m venv venv-tpot
    source ./venv-tpot/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel pandas scikit-learn scikit-multilearn deap update_checker tqdm stopit xgboost torch tpot
    python3.8 ./automl_tpot.py $id
    pkill -f tpot
    sleep 10

    echo Finished processing dataset $id at $(date).

done

# Utils
echo ======== Utils ========
python3.8 -m venv venv-utils
source ./venv-utils/bin/activate
python3.8 -m pip install --upgrade pip
python3.8 -m pip install --upgrade setuptools pytictoc wheel pandas scikit-learn scikit-multilearn ydata-profiling distinctipy matplotlib tabulate openpyxl
python3.8 ./utils_consolidator.py
python3.8 ./utils_plot_f1_scores.py
python3.8 ./utils_plot_training_times.py

echo Script execution finished at $(date).
