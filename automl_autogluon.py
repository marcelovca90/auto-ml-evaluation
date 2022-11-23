
from common import collect_and_persist_results, load_csv, load_openml, DATASET_FOLDER, SEED, TIMER
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd

try:

    X_train, X_test, y_train, y_test = load_openml(44)
    train_df = pd.DataFrame(X_train).assign(**{'class': pd.Series(y_train)}).dropna()
    test_df = pd.DataFrame(X_test).assign(**{'class': pd.Series(y_test)}).dropna()

    clf = TabularPredictor(eval_metric='accuracy', label='class')

    TIMER.tic()
    clf = clf.fit(time_limit=1*60, train_data=train_df)
    training_time = TIMER.tocvalue()

    TIMER.tic()
    y_test = test_df['class'].values
    y_pred = clf.predict(test_df)
    test_time = TIMER.tocvalue()

    collect_and_persist_results(y_test, y_pred, training_time, test_time, "autogluon")

except Exception as e:
    print(f'Cannot run autogluon for dataset {DATASET_FOLDER}. Reason: {str(e)}')
