import pandas as pd
import blobcity as bc
import numpy as np

from common import *

np.bool = np.bool_ # # https://stackoverflow.com/a/76224186/2679529
np.int = np.int_ # https://stackoverflow.com/a/74946903/2679529
np.float = np.float_ # https://stackoverflow.com/a/76322115/2679529

if __name__ == "__main__":

    for SEED in PRIME_NUMBERS:
        
        try:

            set_random_seed(SEED)

            X_train, X_test, y_train, y_test = load_data_delegate(SEED)
            train_df = pd.DataFrame(X_train).assign(**{'class': pd.Series(y_train)}).dropna()
            test_df = pd.DataFrame(X_test).assign(**{'class': pd.Series(y_test)}).dropna()

            TIMER.tic()
            clf = bc.train(df=train_df, target='class')
            training_time = TIMER.tocvalue()

            TIMER.tic()
            y_test = test_df['class'].values
            y_pred = clf.predict(test_df)
            test_time = TIMER.tocvalue()

            collect_and_persist_results(y_test, y_pred, training_time, test_time, "autogluon", SEED)

        except Exception as e:
            print(f'Cannot run autogluon for dataset {get_dataset_ref()} (seed={SEED}). Reason: {str(e)}')
