from common import collect_and_persist_scores, load_csv, load_openml, DATASET_FOLDER, SEED, TIMER
from autosklearn.classification import AutoSklearnClassifier

try:

    X_train, X_test = load_csv(DATASET_FOLDER, 'X_train.csv'), load_csv(DATASET_FOLDER, 'X_test.csv')
    y_train, y_test = load_csv(DATASET_FOLDER, 'y_train.csv'), load_csv(DATASET_FOLDER, 'y_test.csv')

    clf = AutoSklearnClassifier(time_left_for_this_task=60*60, resampling_strategy="cv", resampling_strategy_arguments={"folds": 5}, seed=SEED)

    TIMER.tic()
    clf.fit(X_train, y_train)
    TIMER.toc()

    TIMER.tic()
    y_pred = clf.predict(X_test)
    TIMER.toc()

    collect_and_persist_scores(y_test, y_pred, "autosklearn")

except Exception as e:
    print(f'Cannot run autosklearn for dataset {DATASET_FOLDER}. Reason: {str(e)}')
