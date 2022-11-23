
from common import collect_and_persist_scores, load_csv, load_openml, DATASET_FOLDER, SEED, TIMER
from pycaret.classification import *
import pandas as pd

try:

    X_train, X_test, y_train, y_test = load_openml(44)

    data_train = pd.DataFrame(X_train)
    data_train['class'] = pd.Series(y_train)

    setup(data_train, target='class', silent=True)

    TIMER.tic()
    best = compare_models(budget_time=1, cross_validation=True, fold=5, sort="Accuracy")
    TIMER.toc()

    data_test = pd.DataFrame(X_test)
    data_test['class'] = pd.Series(y_test)

    TIMER.tic()
    y_pred_df = predict_model(best, data_test, raw_score=True)
    y_pred = [np.argmax(i) for i in y_pred_df[["Score_0", "Score_1"]].to_numpy()]
    TIMER.toc()

    collect_and_persist_scores(y_test, y_pred, "pycaret")

except Exception as e:
    print(f'Cannot run pycaret for dataset {DATASET_FOLDER}. Reason: {str(e)}')
