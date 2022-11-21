from common import collect_and_persist_scores, load_openml, SEED, TIMER
from autosklearn.classification import AutoSklearnClassifier

X_train, X_test, y_train, y_test = load_openml()

clf = AutoSklearnClassifier(time_left_for_this_task=10*60, resampling_strategy="cv", resampling_strategy_arguments={"folds": 5}, seed=SEED)

TIMER.tic()
clf.fit(X_train, y_train)
TIMER.toc()

TIMER.tic()
y_pred = clf.predict(X_test)
TIMER.toc()

collect_and_persist_scores(y_test, y_pred, "autosklearn")
