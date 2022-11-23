from common import collect_and_persist_scores, load_csv, load_openml, DATASET_FOLDER, SEED, TIMER
import autokeras as ak

try:

    X_train, X_test = load_csv(DATASET_FOLDER, 'X_train.csv'), load_csv(DATASET_FOLDER, 'X_test.csv')
    y_train, y_test = load_csv(DATASET_FOLDER, 'y_train.csv'), load_csv(DATASET_FOLDER, 'y_test.csv')

    autokeras = ak.StructuredDataClassifier(max_trials=10, overwrite=True, seed=SEED)

    TIMER.tic()
    autokeras.fit(X_train, y_train, epochs=100)
    TIMER.toc()

    TIMER.tic()
    y_pred = autokeras.predict(X_test)
    TIMER.toc()

    collect_and_persist_scores(y_test, y_pred, "autokeras")

except Exception as e:
    print(f'Cannot run autokeras for dataset {DATASET_FOLDER}. Reason: {str(e)}')
