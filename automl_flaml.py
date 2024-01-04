from flaml import AutoML

from common import *

if __name__ == "__main__":

    try:

        for SEED in PRIME_NUMBERS:

            set_random_seed(SEED)

            X_train, X_test, y_train, y_test = load_data_delegate(SEED)

            clf = AutoML(n_jobs=NUM_CPUS, seed=SEED)
            print('yadayadayadayadayadayadayadayadayadayada', type(clf), clf)

            TIMER.tic()
            clf.fit(
                X_train, 
                y_train, 
                metric="accuracy",
                task="classification",
                time_budget=EXEC_TIME_SECONDS
            )
            training_time = TIMER.tocvalue()

            TIMER.tic()
            print('yadayadayadayadayadayadayadayadayadayada', type(clf), clf)
            y_pred = clf.predict(X_test)
            test_time = TIMER.tocvalue()

            collect_and_persist_results(y_test, y_pred, training_time, test_time, "flaml", SEED)

    except Exception as e:
        print(f'Cannot run flaml for dataset {get_dataset_ref()}. Reason: {str(e)}')
