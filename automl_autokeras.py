from common import collect_and_persist_results, load_csv, load_openml, DATASET_FOLDER, SEED, TIMER
import autokeras as ak

try:

    X_train, X_test, y_train, y_test = load_openml(44)

    autokeras = ak.StructuredDataClassifier(max_trials=1, overwrite=True, seed=SEED)

    TIMER.tic()
    autokeras.fit(X_train, y_train, epochs=100)
    training_time = TIMER.tocvalue()

    TIMER.tic()
    y_pred = autokeras.predict(X_test)
    test_time = TIMER.tocvalue()

    collect_and_persist_results(y_test, y_pred, training_time, test_time, "autokeras")

except Exception as e:
    print(f'Cannot run autokeras for dataset {DATASET_FOLDER}. Reason: {str(e)}')
