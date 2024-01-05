from evalml.automl import AutoMLSearch

from common import *

if __name__ == "__main__":

    for SEED in PRIME_NUMBERS:
        
        try:

            set_random_seed(SEED)

            X_train, X_test, y_train, y_test = load_data_delegate(SEED)

            clf = AutoMLSearch(
                X_train=X_train, 
                y_train=y_train, 
                problem_type=infer_task_type(y_test), 
                max_time=EXEC_TIME_SECONDS,
                n_jobs=NUM_CPUS,
                random_seed=SEED
            )

            TIMER.tic()
            clf.search()
            best = clf.best_pipeline.fit(X_train, y_train)
            training_time = TIMER.tocvalue()

            TIMER.tic()
            y_pred = best.predict(X_test)
            test_time = TIMER.tocvalue()

            collect_and_persist_results(y_test, y_pred, training_time, test_time, "evalml", SEED)

        except Exception as e:
            print(f'Cannot run evalml for dataset {get_dataset_ref()} (seed={SEED}). Reason: {str(e)}')
