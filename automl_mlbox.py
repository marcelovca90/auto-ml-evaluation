
from common import collect_and_persist_results, load_csv, load_openml, DATASET_FOLDER, SEED, TIMER
from mlbox.optimisation import *
from mlbox.prediction import *
import pandas as pd

try:

    X_train, X_test, y_train, y_test = load_openml(44)

    data_train = {"train": pd.DataFrame(X_train), "target": pd.Series(y_train)}

    opt = Optimiser()

    TIMER.tic()
    opt.evaluate(None, data_train)
    params = opt.optimise(None, data_train)
    TIMER.toc()

    data_test = {"train": pd.DataFrame(X_train), "test": pd.DataFrame(X_test), "target": pd.Series(y_train)}

    pred = Predictor()

    TIMER.tic()
    y_pred = pred.fit_predict(params, data_test)
    TIMER.toc()

except Exception as e:
    print(f'Cannot run mlbox for dataset {DATASET_FOLDER}. Reason: {str(e)}')
