import pandas as pd
from h2o.sklearn import H2OAutoMLClassifier

from common import *

if __name__ == "__main__":

    for SEED in PRIME_NUMBERS:
        
        try:

            set_random_seed(SEED)

            X_train, X_test, y_train, y_test = load_data_delegate(SEED)
            train_df = pd.DataFrame(X_train).assign(**{'class': pd.Series(y_train)}).dropna()
            test_df = pd.DataFrame(X_test).assign(**{'class': pd.Series(y_test)}).dropna()

            clf = H2OAutoMLClassifier(
                max_runtime_secs=EXEC_TIME_SECONDS, 
                sort_metric='auto', 
                nfolds=5, 
                seed=SEED
            )

            TIMER.tic()
            clf.fit(train_df.drop('class', axis=1).values, train_df['class'].values)
            training_time = TIMER.tocvalue()

            TIMER.tic()
            y_pred = clf.predict(X_test)
            test_time = TIMER.tocvalue()

            collect_and_persist_results(y_test, y_pred, training_time, test_time, "h2o", SEED)

        except Exception as e:
            print(f'Cannot run h2o for dataset {get_dataset_ref()} (seed={SEED}). Reason: {str(e)}')
