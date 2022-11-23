from common import collect_and_persist_scores, load_csv, load_openml, DATASET_FOLDER, SEED, TIMER
from tpot import TPOTClassifier

try:

    X_train, X_test, y_train, y_test = load_openml(44)

    clf = TPOTClassifier(max_time_mins=1, cv=5, random_state=SEED)

    TIMER.tic()
    clf.fit(X_train, y_train)
    TIMER.toc()

    TIMER.tic()
    y_pred = clf.predict(X_test)
    TIMER.toc()

    collect_and_persist_scores(y_test, y_pred, "tpot")

except Exception as e:
    print(f'Cannot run tpot for dataset {DATASET_FOLDER}. Reason: {str(e)}')
