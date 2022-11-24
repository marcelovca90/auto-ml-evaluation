import numpy as np
import pandas as pd
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

from common import *

try:

    X_train, X_test, y_train, y_test = load_data_delegate()
    train_df = pd.DataFrame(X_train).assign(**{'class': pd.Series(y_train)}).dropna()
    test_df = pd.DataFrame(X_test).assign(**{'class': pd.Series(y_test)}).dropna()

    clf = TabularAutoML(task=Task(infer_task_type(y_test), metric='accuracy'), timeout=EXEC_TIME_SECONDS)

    TIMER.tic()
    clf.fit_predict(train_df, roles={'target': 'class'})
    training_time = TIMER.tocvalue()

    TIMER.tic()
    y_test = test_df['class']
    y_pred = np.argmax(clf.predict(test_df).data, axis=1)
    test_time = TIMER.tocvalue()

    collect_and_persist_results(y_test, y_pred, training_time, test_time, "lightautoml")

except Exception as e:
    print(f'Cannot run lightautoml for dataset {DATASET_REF}. Reason: {str(e)}')
