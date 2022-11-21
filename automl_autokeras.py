from common import collect_and_persist_scores, load_openml, SEED, TIMER
import autokeras as ak

X_train, X_test, y_train, y_test = load_openml()
autokeras = ak.StructuredDataClassifier(max_trials=10, overwrite=True)

TIMER.tic()
autokeras.fit(X_train, y_train, epochs=100)
TIMER.toc()

TIMER.tic()
y_pred = autokeras.predict(X_test)
TIMER.toc()

collect_and_persist_scores(y_test, y_pred, "autokeras")