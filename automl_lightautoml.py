from common import collect_and_persist_scores, load_csv, load_openml, DATASET_FOLDER, SEED, TIMER
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
import pandas as pd

try:

    X_train, X_test, y_train, y_test = load_openml(44)
    train_df = pd.DataFrame(X_train).assign(**{'class': pd.Series(y_train)}).dropna()
    test_df = pd.DataFrame(X_test).assign(**{'class': pd.Series(y_test)}).dropna()

    clf = TabularAutoML(task=Task('binary', metric='accuracy'), timeout=1*60)

    TIMER.tic()
    clf.fit_predict(train_df, roles={'target': 'class'})
    TIMER.toc()

    TIMER.tic()
    y_test = test_df['class'].values
    y_pred = (clf.predict(test_df).data[:, 0] > 0.5).astype(int)
    TIMER.toc()

    collect_and_persist_scores(y_test, y_pred, "autolightml")

except Exception as e:
    print(f'Cannot run autolightml for dataset {DATASET_FOLDER}. Reason: {str(e)}')
