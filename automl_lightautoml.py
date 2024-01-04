import numpy as np
import pandas as pd
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

from common import *

if __name__ == "__main__":
    
    try:

        for SEED in PRIME_NUMBERS:

            set_random_seed(SEED)

            X_train, X_test, y_train, y_test = load_data_delegate(SEED)
            train_df = pd.DataFrame(X_train).assign(**{'class': pd.Series(y_train)}).dropna()
            train_df = train_df.rename(columns={i:str(i) for i in train_df.columns})
            test_df = pd.DataFrame(X_test).assign(**{'class': pd.Series(y_test)}).dropna()
            test_df = test_df.rename(columns={i:str(i) for i in test_df.columns})

            clf = TabularAutoML(
                task=Task(infer_task_type(y_test), metric='accuracy'),
                timeout=EXEC_TIME_SECONDS,
                cpu_limit=NUM_CPUS,
                reader_params = {'cv': 5, 'n_jobs': NUM_CPUS, 'random_state': SEED},
            )

            TIMER.tic()
            feature_names = [str(i) for i in train_df.columns]
            clf.fit_predict(train_df, roles={'target': 'class'})
            training_time = TIMER.tocvalue()

            TIMER.tic()
            y_test = test_df['class']
            y_pred = np.argmax(clf.predict(test_df).data, axis=1)
            test_time = TIMER.tocvalue()

            collect_and_persist_results(y_test, y_pred, training_time, test_time, "lightautoml", SEED)

    except Exception as e:
        print(f'Cannot run lightautoml for dataset {get_dataset_ref()}. Reason: {str(e)}')
