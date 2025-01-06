from multiprocessing import freeze_support
from autoPyTorch.api.tabular_classification import TabularClassificationTask

from common import *

if __name__ == "__main__":

    for SEED in PRIME_NUMBERS:
        
        try:

            set_random_seed(SEED)

            X_train, X_test, y_train, y_test = load_data_delegate(SEED)

            clf = TabularClassificationTask(n_jobs=NUM_CPUS, seed=SEED)

            TIMER.tic()
            clf.search(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                optimize_metric='f1_weighted',
                budget_type='runtime',
                total_walltime_limit=EXEC_TIME_SECONDS,
                func_eval_time_limit_secs=EXEC_TIME_SECONDS//10,
                memory_limit=MAX_MEMORY_MB
            )
            training_time = TIMER.tocvalue()

            TIMER.tic()
            y_pred = clf.predict(X_test)
            test_time = TIMER.tocvalue()

            collect_and_persist_results(y_test, y_pred, training_time, test_time, "autopytorch", SEED)

        except Exception as e:
            print(f'Cannot run autopytorch for dataset {get_dataset_ref()} (seed={SEED}). Reason: {str(e)}')
