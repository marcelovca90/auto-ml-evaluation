from common import collect_and_persist_scores, load_csv, load_openml, DATASET_FOLDER, SEED, TIMER
from h2o.sklearn import H2OAutoMLClassifier

try:

    X_train, X_test, y_train, y_test = load_openml(44)

    clf = H2OAutoMLClassifier(max_runtime_secs=1*60, nfolds=5, seed=SEED, sort_metric='accuracy')

    TIMER.tic()
    clf.fit(X_train, y_train)
    TIMER.toc()

    TIMER.tic()
    y_pred = clf.predict(X_test)
    TIMER.toc()

    collect_and_persist_scores(y_test, y_pred, "h2o")

except Exception as e:
    print(f'Cannot run h2o for dataset {DATASET_FOLDER}. Reason: {str(e)}')
