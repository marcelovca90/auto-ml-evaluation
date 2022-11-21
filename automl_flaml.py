from common import collect_and_persist_scores, load_openml, SEED, TIMER
from flaml import AutoML

X_train, X_test, y_train, y_test = load_openml()

clf = AutoML()

TIMER.tic()
clf.fit(X_train, y_train, task="classification", time_budget=1*10)
TIMER.toc()

TIMER.tic()
y_pred = clf.predict(X_test)
TIMER.toc()

collect_and_persist_scores(y_test, y_pred, "flaml")
