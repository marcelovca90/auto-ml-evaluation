from tpot import TPOTClassifier

from common import *

try:

    X_train, X_test, y_train, y_test = load_data_delegate()

    clf = TPOTClassifier(max_time_mins=EXEC_TIME_MINUTES, cv=5, random_state=SEED)

    TIMER.tic()
    clf.fit(X_train, y_train)
    training_time = TIMER.tocvalue()

    TIMER.tic()
    y_pred = clf.predict(X_test)
    test_time = TIMER.tocvalue()

    collect_and_persist_results(y_test, y_pred, training_time, test_time, "tpot")

except Exception as e:
    print(f'Cannot run tpot for dataset {DATASET_REF}. Reason: {str(e)}')
