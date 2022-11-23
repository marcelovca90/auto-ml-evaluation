from common import collect_and_persist_scores, load_csv, load_openml, DATASET_FOLDER, SEED, TIMER
from autoPyTorch.api.tabular_classification import TabularClassificationTask

try:

    X_train, X_test, y_train, y_test = load_openml(44)

    clf = TabularClassificationTask(seed=SEED)

    TIMER.tic()
    clf.search(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        optimize_metric='accuracy',
        total_walltime_limit=1*60,
        func_eval_time_limit_secs=1*60,
        memory_limit=8192
    )
    TIMER.toc()

    TIMER.tic()
    y_pred = clf.predict(X_test)
    TIMER.toc()

    collect_and_persist_scores(y_test, y_pred, "autopytorch")

except Exception as e:
    print(f'Cannot run autopytorch for dataset {DATASET_FOLDER}. Reason: {str(e)}')
