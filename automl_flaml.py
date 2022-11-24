from flaml import AutoML

from common import (DATASET_FOLDER, EXEC_TIME_MINUTES, EXEC_TIME_SECONDS, SEED,
                    TIMER, collect_and_persist_results, load_csv, load_openml)

try:

    X_train, X_test, y_train, y_test = load_openml(44)

    clf = AutoML()

    TIMER.tic()
    clf.fit(X_train, y_train, metric="accuracy", task="classification", time_budget=EXEC_TIME_SECONDS)
    training_time = TIMER.tocvalue()

    TIMER.tic()
    y_pred = clf.predict(X_test)
    test_time = TIMER.tocvalue()

    collect_and_persist_results(y_test, y_pred, training_time, test_time, "flaml")

except Exception as e:
    print(f'Cannot run flaml for dataset {DATASET_FOLDER}. Reason: {str(e)}')
