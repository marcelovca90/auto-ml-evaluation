from autosklearn.classification import AutoSklearnClassifier

from common import *

if __name__ == "__main__":

    for SEED in PRIME_NUMBERS:
        
        try:

            set_random_seed(SEED)
                
            X_train, X_test, y_train, y_test = load_data_delegate(SEED)

            clf = AutoSklearnClassifier(
                time_left_for_this_task=EXEC_TIME_SECONDS,
                per_run_time_limit=EXEC_TIME_SECONDS//10,
                resampling_strategy="cv",
                resampling_strategy_arguments={"folds": 5},
                n_jobs=NUM_CPUS,
                seed=SEED,
                memory_limit=MAX_MEMORY_MB
            )

            TIMER.tic()
            clf.fit(X_train, y_train)
            training_time = TIMER.tocvalue()

            TIMER.tic()
            y_pred = clf.predict(X_test)
            test_time = TIMER.tocvalue()

            collect_and_persist_results(y_test, y_pred, training_time, test_time, "autosklearn", SEED)

        except Exception as e:
            print(f'Cannot run autosklearn for dataset {get_dataset_ref()} (seed={SEED}). Reason: {str(e)}')
