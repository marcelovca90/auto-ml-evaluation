import pandas as pd
from fedot import Fedot

from common import *

if __name__ == "__main__":

    for SEED in PRIME_NUMBERS:
        
        try:

            set_random_seed(SEED)

            X_train, X_test, y_train, y_test = load_data_delegate(SEED)

            clf = Fedot(timeout=EXEC_TIME_MINUTES, problem='classification', seed=SEED)

            TIMER.tic()
            clf.fit(X_train, y_train)
            training_time = TIMER.tocvalue()

            TIMER.tic()
            y_pred = clf.predict(X_test)
            test_time = TIMER.tocvalue()

            collect_and_persist_results(y_test, y_pred, training_time, test_time, "fedot", SEED)

        except Exception as e:
            print(f'Cannot run fedot for dataset {get_dataset_ref()} (seed={SEED}). Reason: {str(e)}')
