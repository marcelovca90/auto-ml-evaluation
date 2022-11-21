from common import collect_and_persist_scores, load_openml, SEED, TIMER
import h2o
from h2o.sklearn import H2OAutoMLClassifier

X_train, X_test, y_train, y_test = load_openml()

clf = H2OAutoMLClassifier(max_models=10, seed=SEED, sort_metric='accuracy')

TIMER.tic()
clf.fit(X_train, y_train)
TIMER.toc()

TIMER.tic()
y_pred = clf.predict(X_test)
TIMER.toc()

collect_and_persist_scores(y_test, y_pred, "h2o")