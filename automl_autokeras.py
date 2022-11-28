import autokeras as ak

from common import *

try:

    X_train, X_test, y_train, y_test = load_data_delegate()

    autokeras = ak.StructuredDataClassifier(multi_label=True, max_trials=3, overwrite=True, seed=SEED)

    TIMER.tic()
    autokeras.fit(X_train, y_train, epochs=1000)
    training_time = TIMER.tocvalue()

    TIMER.tic()
    # multilabel datasets
    if get_dataset_ref() in [41465, 41468, 42470, 41471, 41473]:
        y_pred = autokeras.predict(X_test).astype(int)
    # binary and multiclass datasets
    else:
        y_pred = autokeras.predict(X_test).astype(int).flatten()
    test_time = TIMER.tocvalue()

    collect_and_persist_results(y_test, y_pred, training_time, test_time, "autokeras")

except Exception as e:
    print(f'Cannot run autokeras for dataset {get_dataset_ref()}. Reason: {str(e)}')
