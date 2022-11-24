import pandas as pd
from h2o.sklearn import H2OAutoMLClassifier

from common import *

try:

    X_train, X_test, y_train, y_test = load_data_delegate()
    train_df = pd.DataFrame(X_train).assign(**{'class': pd.Series(y_train)}).dropna()
    test_df = pd.DataFrame(X_test).assign(**{'class': pd.Series(y_test)}).dropna()

    clf = H2OAutoMLClassifier(max_runtime_secs=EXEC_TIME_SECONDS, nfolds=5, seed=SEED, sort_metric='accuracy')

    TIMER.tic()
    clf.fit(train_df.drop('class', axis=1).values, train_df['class'].values)
    training_time = TIMER.tocvalue()

    TIMER.tic()
    y_pred = clf.predict(X_test)
    test_time = TIMER.tocvalue()

    collect_and_persist_results(y_test, y_pred, training_time, test_time, "h2o")

except Exception as e:
    print(f'Cannot run h2o for dataset {DATASET_REF}. Reason: {str(e)}')
