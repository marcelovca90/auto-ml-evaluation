from autosklearn.classification import AutoSklearnClassifier

from common import *

try:

    X_train, X_test, y_train, y_test = load_data_delegate()

    clf = AutoSklearnClassifier(time_left_for_this_task=EXEC_TIME_SECONDS, resampling_strategy="cv", resampling_strategy_arguments={"folds": 5}, seed=SEED)

    TIMER.tic()
    clf.fit(X_train, y_train)
    training_time = TIMER.tocvalue()

    TIMER.tic()
    y_pred = clf.predict(X_test)
    test_time = TIMER.tocvalue()

    collect_and_persist_results(y_test, y_pred, training_time, test_time, "autosklearn")

except Exception as e:
    print(f'Cannot run autosklearn for dataset {get_dataset_ref()}. Reason: {str(e)}')
