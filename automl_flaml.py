from common import collect_and_persist_scores, load_csv, load_openml, DATASET_FOLDER, SEED, TIMER
from flaml import AutoML

try:

    X_train, X_test, y_train, y_test = load_openml(44)

    clf = AutoML()

    TIMER.tic()
    clf.fit(X_train, y_train, metric="accuracy", task="classification", time_budget=1*60)
    TIMER.toc()

    TIMER.tic()
    y_pred = clf.predict(X_test)
    TIMER.toc()

    collect_and_persist_scores(y_test, y_pred, "flaml")

except Exception as e:
    print(f'Cannot run flaml for dataset {DATASET_FOLDER}. Reason: {str(e)}')
