from common import collect_and_persist_scores, load_csv, load_openml, DATASET_FOLDER, SEED, TIMER
from evalml.automl import AutoMLSearch

try:

    X_train, X_test, y_train, y_test = load_openml(44)

    clf = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type='binary', random_seed=SEED, max_time=1*60)

    TIMER.tic()
    clf.search()
    best = clf.best_pipeline.fit(X_train, y_train)
    TIMER.toc()

    TIMER.tic()
    y_pred = best.predict(X_test)
    TIMER.toc()

    collect_and_persist_scores(y_test, y_pred, "evalml")

except Exception as e:
    print(f'Cannot run evalml for dataset {DATASET_FOLDER}. Reason: {str(e)}')
