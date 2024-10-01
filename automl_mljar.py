import pandas as pd
from supervised.automl import AutoML

from common import *

if __name__ == "__main__":

    for SEED in PRIME_NUMBERS:
        
        try:

            set_random_seed(SEED)

            X_train, X_test, y_train, y_test = load_data_delegate(SEED)

            ml_task = f'{infer_task_type(y_test)}_classification'
            clf = AutoML(total_time_limit=EXEC_TIME_SECONDS, ml_task=ml_task, random_state=SEED)

            TIMER.tic()
            clf = clf.fit(X_train, y_train)
            training_time = TIMER.tocvalue()

            TIMER.tic()
            y_pred = clf.predict(X_test)
            test_time = TIMER.tocvalue()

            collect_and_persist_results(y_test, y_pred, training_time, test_time, "mljar", SEED)

        except Exception as e:
            print(f'Cannot run mljar for dataset {get_dataset_ref()} (seed={SEED}). Reason: {str(e)}')
