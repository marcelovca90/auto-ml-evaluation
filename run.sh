#!/bin/bash

echo Script execution started at $(date).

# Preparation
echo ======== Preparation ========
rm *.log &> /dev/null
rm -rf venv-* &> /dev/null
rm -rf structured* &> /dev/null
rm -rf Autogluon* &> /dev/null
rm results/* &> /dev/null

# AutoGluon
echo ======== AutoGluon ========
python3.8 -m venv venv-autogluon
source ./venv-autogluon/bin/activate
python3.8 -m pip install --upgrade pip
python3.8 -m pip install --upgrade setuptools pytictoc wheel torch==1.12+cpu torchvision==0.13.0+cpu torchtext==0.13.0 -f https://download.pytorch.org/whl/cpu/torch_stable.html autogluon
python3.8 ./automl_autogluon.py
 
# AutoKeras
echo ======== AutoKeras ========
python3.8 -m venv venv-autokeras
source ./venv-autokeras/bin/activate
python3.8 -m pip install --upgrade pip
python3.8 -m pip install --upgrade setuptools pytictoc wheel git+https://github.com/keras-team/keras-tuner.git scikit-learn autokeras
python3.8 ./automl_autokeras.py

# AutoPyTorch
echo ======== AutoPyTorch ========
python3.8 -m venv venv-autopytorch
source ./venv-autopytorch/bin/activate
python3.8 -m pip install --upgrade pip
python3.8 -m pip install --upgrade setuptools pytictoc wheel swig torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu autoPyTorch
python3.8 ./automl_autopytorch.py

# AutoSklearn
echo ======== AutoSklearn ========
python3.8 -m venv venv-autosklearn
source ./venv-autosklearn/bin/activate
python3.8 -m pip install --upgrade pip
python3.8 -m pip install --upgrade setuptools pytictoc wheel auto-sklearn
python3.8 ./automl_autosklearn.py

# EvalML
echo ======== EvalML ========
python3.8 -m venv venv-evalml
source ./venv-evalml/bin/activate
python3.8 -m pip install --upgrade pip
python3.8 -m pip install --upgrade setuptools pytictoc wheel evalml
python3.8 ./automl_evalml.py

# FLAML
echo ======== FLAML ========
python3.8 -m venv venv-flaml
source ./venv-flaml/bin/activate
python3.8 -m pip install --upgrade pip
python3.8 -m pip install --upgrade setuptools pytictoc wheel flaml
python3.8 ./automl_flaml.py

# GAMA
echo ======== GAMA ========
python3.8 -m venv venv-gama
source ./venv-gama/bin/activate
python3.8 -m pip install --upgrade pip
python3.8 -m pip install --upgrade setuptools pytictoc wheel gama
python3.8 ./automl_gama.py

# H2O
echo ======== H2O ========
python3.8 -m venv venv-h2o
source ./venv-h2o/bin/activate
python3.8 -m pip install --upgrade pip
python3.8 -m pip install --upgrade setuptools pytictoc wheel requests tabulate future scikit-learn pandas h2o
python3.8 ./automl_h2o.py

# LightAutoML
echo ======== LightAutoML ========
python3.8 -m venv venv-lightautoml
source ./venv-lightautoml/bin/activate
python3.8 -m pip install --upgrade pip
python3.8 -m pip install --upgrade setuptools pytictoc wheel lightautoml
python3.8 ./automl_lightautoml.py

# TPOT
echo ======== TPOT ========
python3.8 -m venv venv-tpot
source ./venv-tpot/bin/activate
python3.8 -m pip install --upgrade pip
python3.8 -m pip install --upgrade setuptools pytictoc wheel deap update_checker tqdm stopit xgboost torch tpot
python3.8 ./automl_tpot.py

# Plotter
echo ======== Plotter ========
python3.8 -m venv venv-plotter
source ./venv-plotter/bin/activate
python3.8 -m pip install --upgrade pip
python3.8 -m pip install --upgrade setuptools pytictoc wheel scikit-learn matplotlib pandas tabulate
python3.8 ./plotter.py

echo Script execution finished at $(date).
