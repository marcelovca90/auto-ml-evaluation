#!/bin/bash

echo Script execution started.

# FLAML
echo ---- FLAML ----
python3.8 -m venv venv-flaml
source ./venv-flaml/bin/activate
python3.8 -m pip install --upgrade pip
python3.8 -m pip install --upgrade setuptools pytictoc flaml
python3.8 ./automl_flaml.py

# AutoSklearn
echo ---- AutoSklearn ----
python3.8 -m venv venv-autosklearn
source ./venv-autosklearn/bin/activate
python3.8 -m pip install --upgrade pip
python3.8 -m pip install --upgrade setuptools pytictoc auto-sklearn
python3.8 ./automl_autosklearn.py

# TPOT
echo ---- TPOT ----
python3.8 -m venv venv-tpot
source ./venv-tpot/bin/activate
python3.8 -m pip install --upgrade pip
python3.8 -m pip install --upgrade setuptools pytictoc deap update_checker tqdm stopit xgboost tpot
python3.8 ./automl_tpot.py

# H2O
echo ---- H2O ----
python3.8 -m venv venv-h2o
source ./venv-h2o/bin/activate
python3.8 -m pip install --upgrade pip
python3.8 -m pip install --upgrade setuptools pytictoc requests tabulate future scikit-learn pandas h2o
python3.8 ./automl_h2o.py

# AutoPyTorch
echo ---- AutoPyTorch ----
python3.8 -m venv venv-autopytorch
source ./venv-autopytorch/bin/activate
python3.8 -m pip install --upgrade pip
python3.8 -m pip install --upgrade setuptools pytictoc swig torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu autoPyTorch
python3.8 ./automl_autopytorch.py

# # AutoKeras
# echo ---- AutoKeras ----
# python3.8 -m venv venv-autokeras
# source ./venv-autokeras/bin/activate
# python3.8 -m pip install --upgrade pip
# python3.8 -m pip install --upgrade setuptools pytictoc git+https://github.com/keras-team/keras-tuner.git scikit-learn autokeras
# python3.8 ./automl_autokeras.py

# Plotter
echo ---- Plotter ----
python3.8 -m venv venv-plotter
source ./venv-plotter/bin/activate
python3.8 -m pip install --upgrade pip
python3.8 -m pip install --upgrade setuptools pytictoc matplotlib
python3.8 ./plotter.py

echo Script execution finished.