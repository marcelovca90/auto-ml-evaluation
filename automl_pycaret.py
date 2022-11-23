
from common import collect_and_persist_results, load_csv, load_openml, DATASET_FOLDER, SEED, TIMER
from pycaret.classification import *
import pandas as pd

try:

    X_train, X_test, y_train, y_test = load_openml(44)
    train_df = pd.DataFrame(X_train).assign(**{'class': pd.Series(y_train)}).dropna()
    test_df = pd.DataFrame(X_test).assign(**{'class': pd.Series(y_test)}).dropna()

    setup(train_df, target='class', silent=True)

    TIMER.tic()
    best = compare_models(budget_time=1, cross_validation=True, fold=5, sort="Accuracy")
    training_time = TIMER.tocvalue()

    data_test = pd.DataFrame(X_test)
    data_test['class'] = pd.Series(y_test)

    TIMER.tic()
    y_test = test_df['class'].values
    y_pred_df = predict_model(best, test_df, raw_score=True)
    y_pred = [np.argmax(i) for i in y_pred_df[["Score_0", "Score_1"]].to_numpy()]
    test_time = TIMER.tocvalue()

    collect_and_persist_results(y_test, y_pred, training_time, test_time, "pycaret")

except Exception as e:
    print(f'Cannot run pycaret for dataset {DATASET_FOLDER}. Reason: {str(e)}')
