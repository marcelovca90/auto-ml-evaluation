import pandas as pd
from naiveautoml import NaiveAutoML

from common import *

if __name__ == "__main__":

    for SEED in PRIME_NUMBERS:
        
        try:

            set_random_seed(SEED)

            X_train, X_test, y_train, y_test = load_data_delegate(SEED)

            task_type = 'multilabel-indicator' if (is_multi_label() and not LABEL_POWERSET) else 'classification'
            clf = NaiveAutoML(timeout=EXEC_TIME_SECONDS,
                              execution_timeout=EXEC_TIME_SECONDS//10,
                              task_type=task_type,
                              scoring='accuracy'
            )

            TIMER.tic()
            clf.fit(X_train, y_train)
            training_time = TIMER.tocvalue()

            TIMER.tic()
            y_pred = clf.predict(X_test)
            test_time = TIMER.tocvalue()

            collect_and_persist_results(y_test, y_pred, training_time, test_time, "naive", SEED)

        except Exception as e:
            print(f'Cannot run naive for dataset {get_dataset_ref()} (seed={SEED}). Reason: {str(e)}')
