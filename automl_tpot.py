from common import collect_and_persist_results, load_csv, load_openml, DATASET_FOLDER, SEED, TIMER
from tpot import TPOTClassifier

try:

    X_train, X_test, y_train, y_test = load_openml(44)

    clf = TPOTClassifier(max_time_mins=1, cv=5, random_state=SEED)

    TIMER.tic()
    clf.fit(X_train, y_train)
    training_time = TIMER.tocvalue()

    TIMER.tic()
    y_pred = clf.predict(X_test)
    test_time = TIMER.tocvalue()

    collect_and_persist_results(y_test, y_pred, training_time, test_time, "tpot")

except Exception as e:
    print(f'Cannot run tpot for dataset {DATASET_FOLDER}. Reason: {str(e)}')
