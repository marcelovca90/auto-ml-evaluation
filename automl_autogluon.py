import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

from common import *

if __name__ == "__main__":

    try:

        for SEED in PRIME_NUMBERS:

            set_random_seed(SEED)

            X_train, X_test, y_train, y_test = load_data_delegate(SEED)
            train_df = pd.DataFrame(X_train).assign(**{'class': pd.Series(y_train)}).dropna()
            test_df = pd.DataFrame(X_test).assign(**{'class': pd.Series(y_test)}).dropna()

            clf = TabularPredictor(eval_metric='accuracy', label='class')

            TIMER.tic()
            clf = clf.fit(time_limit=EXEC_TIME_SECONDS, train_data=train_df, num_cpus=NUM_CPUS)
            training_time = TIMER.tocvalue()

            TIMER.tic()
            y_test = test_df['class'].values
            y_pred = clf.predict(test_df)
            test_time = TIMER.tocvalue()

            collect_and_persist_results(y_test, y_pred, training_time, test_time, "autogluon", SEED)

    except Exception as e:
        print(f'Cannot run autogluon for dataset {get_dataset_ref()}. Reason: {str(e)}')
