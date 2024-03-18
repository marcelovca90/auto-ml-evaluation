#!/bin/bash

# OpenML datasets
# - binary:
#   - 37        diabetes                        768x9x2
#   - 44        spambase                        4601x58x2
#   - 1462      banknote-authentication         1372x6x2
#   - 1479      hill-valley                     1212x101x2
#   - 1510      wdbc                            569x31x2
# - multiclass:
#   - 23        contraceptive-method-choice     1473x10x10
#   - 181       yeast                           1484x9x10
#   - 1466      cardiotocography                2126x24x10
#   - 40691     wine-quality                    1599x12x6
#   - 40975     car                             1728x7x4
# - multilabel:
#   - 41465     emotions                        593x78x6
#   - 41468     image                           2000x140x5
#   - 41470     reuters                         2000x250x7
#   - 41471     scene                           2407x300x6
#   - 41473     yeast                           2417x117x14

datasets=(37 44 1462 1479 1510 23 181 1466 40691 40975 41465 41468 41470 41471 41473 41465ps 41468ps 41470ps 41471ps 41473ps)

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
echo Finished cleaning files from previous executions at $(date).

for id in ${datasets[@]}; do

    echo ======== Processing ========
    echo Started processing dataset $id at $(date).

    # AutoGluon
    echo ======== AutoGluon ========
    python3.8 -m venv venv-autogluon
    source ./venv-autogluon/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel pandas scikit-learn scikit-multilearn torch==1.12+cpu torchvision==0.13.0+cpu torchtext==0.13.0 -f https://download.pytorch.org/whl/cpu/torch_stable.html autogluon
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
    python3.8 -m pip install --upgrade setuptools pytictoc wheel pandas scikit-learn scikit-multilearn swig torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu autoPyTorch
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

    # PyCaret
    echo ======== PyCaret ========
    python3.8 -m venv venv-pycaret
    source ./venv-pycaret/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel pandas scikit-learn scikit-multilearn pycaret[full]
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
python3.8 -m pip install --upgrade setuptools pytictoc wheel pandas scikit-learn scikit-multilearn matplotlib tabulate openpyxl
python3.8 ./utils_consolidator.py
python3.8 ./utils_plot_f1_scores.py
python3.8 ./utils_plot_training_times.py

echo Script execution finished at $(date).
