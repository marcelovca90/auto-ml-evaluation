from gama import GamaClassifier

from common import *

if __name__ == "__main__":

    try:

        for SEED in PRIME_NUMBERS:

            set_random_seed(SEED)

            X_train, X_test, y_train, y_test = load_data_delegate(SEED)

            clf = GamaClassifier(
                max_total_time=EXEC_TIME_SECONDS, 
                max_eval_time=EXEC_TIME_SECONDS//10,
                store="nothing",
                n_jobs=NUM_CPUS,
                random_state=SEED
            )

            TIMER.tic()
            clf.fit(X_train, y_train)
            training_time = TIMER.tocvalue()

            TIMER.tic()
            y_pred = clf.predict(X_test)
            test_time = TIMER.tocvalue()

            collect_and_persist_results(y_test, y_pred, training_time, test_time, "gama", SEED)

    except Exception as e:
        print(f'Cannot run gama for dataset {get_dataset_ref()}. Reason: {str(e)}')
