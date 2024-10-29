import pandas as pd
from naiveautoml import NaiveAutoML

from common import *

if __name__ == "__main__":

    for SEED in PRIME_NUMBERS:
        
        try:

            set_random_seed(SEED)

            X_train, X_test, y_train, y_test = load_data_delegate(SEED)
            print(y_test)
            print('------------------------->', is_multi_label())
            if is_multi_label():
                train_df = pd.concat([X_train, y_train], axis='columns')
                test_df = pd.concat([X_test, y_test], axis='columns')
                x_cols = train_df.drop(columns=y_test.columns.tolist()).columns.tolist()
                y_cols = y_test.columns.tolist()
                print('------------------------->', 'x_cols', x_cols)
                print('------------------------->', 'y_cols', y_cols)
                
            else:
                train_df = pd.DataFrame(X_train).assign(**{'class': pd.Series(y_train)}).dropna()
                test_df = pd.DataFrame(X_test).assign(**{'class': pd.Series(y_test)}).dropna()

            task_type = 'multilabel-indicator' if (is_multi_label() and not LABEL_POWERSET) else 'classification'
            print('------------------------->', task_type)
            clf = NaiveAutoML(timeout_overall=EXEC_TIME_SECONDS,
                              timeout_candidate=EXEC_TIME_SECONDS//10,
                              task_type=task_type,
                              random_state=SEED
            )

            TIMER.tic()
            clf.fit(train_df[x_cols], train_df[y_cols])
            training_time = TIMER.tocvalue()

            TIMER.tic()
            y_test = test_df[y_cols].values
            y_pred = clf.predict(test_df.drop(columns=y_cols))
            test_time = TIMER.tocvalue()

            collect_and_persist_results(y_test, y_pred, training_time, test_time, "naive", SEED)

        except Exception as e:
            print(f'Cannot run naive for dataset {get_dataset_ref()} (seed={SEED}). Reason: {str(e)}')
