import pandas as pd
from vowpalwabbit.sklearn import VWClassifier, VWMultiClassifier

from common import *

run_script = \
"""
    # VowpalWabbit
    echo ======== VowpalWabbit ========
    python3.8 -m venv venv-vowpal
    source ./venv-vowpal/bin/activate
    python3.8 -m pip install --upgrade pip
    python3.8 -m pip install --upgrade setuptools pytictoc wheel pandas scikit-learn scikit-multilearn vowpalwabbit
    python3.8 ./automl_vowpal.py $id
    pkill -f vowpal
    sleep 10
"""

if __name__ == "__main__":

    for SEED in PRIME_NUMBERS:
        
        try:

            set_random_seed(SEED)

            X_train, X_test, y_train, y_test = load_data_delegate(SEED)
            X_train, X_test, y_train, y_test = \
                X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

            # Ensure sample sizes match
            if X_train.shape[0] != y_train.shape[0]:
                raise ValueError(f"Mismatch in number of samples: X_train has {X_train.shape[0]} samples, "
                                 f"but y_train has {y_train.shape[0]} samples.")
            if X_test.shape[0] != y_test.shape[0]:
                raise ValueError(f"Mismatch in number of samples: X_test has {X_test.shape[0]} samples, "
                                 f"but y_test has {y_test.shape[0]} samples.")

            # Classifier selection based on task type
            if is_multi_label() and not LABEL_POWERSET:
                clf = VWMultiClassifier(oaa=y_train.shape[1], hash_seed=SEED, random_seed=SEED)
            elif infer_task_type(y_test) == 'multiclass':
                clf = VWMultiClassifier(hash_seed=SEED, random_seed=SEED)
            else:
                clf = VWClassifier(hash_seed=SEED, random_seed=SEED)

            TIMER.tic()
            clf.fit(X_train, y_train)
            training_time = TIMER.tocvalue()

            TIMER.tic()
            y_pred = clf.predict(X_test)
            test_time = TIMER.tocvalue()

            collect_and_persist_results(y_test, y_pred, training_time, test_time, "vowpal", SEED)

        except Exception as e:
            print(f'Cannot run vowpal for dataset {get_dataset_ref()} (seed={SEED}). Reason: {str(e)}')
