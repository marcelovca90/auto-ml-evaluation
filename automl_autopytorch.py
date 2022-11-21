from common import collect_and_persist_scores, load_openml, SEED, TIMER
from autoPyTorch.api.tabular_classification import TabularClassificationTask

X_train, X_test, y_train, y_test = load_openml()

clf = TabularClassificationTask()

TIMER.tic()
clf.search(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    optimize_metric='accuracy',
    total_walltime_limit=10*60,
    func_eval_time_limit_secs=60
)
TIMER.toc()

TIMER.tic()
y_pred = clf.predict(X_test)
TIMER.toc()

collect_and_persist_scores(y_test, y_pred, "autopytorch")