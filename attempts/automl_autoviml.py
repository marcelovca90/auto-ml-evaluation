import pandas as pd
from autoviml.Auto_ViML import Auto_ViML

from common import *

if __name__ == "__main__":

    for SEED in PRIME_NUMBERS:
        
        try:

            set_random_seed(SEED)

            X_train, X_test, y_train, y_test = load_data_delegate(SEED)

            feature_names = pd.concat([X_train, X_test]).columns.tolist()
            target = pd.concat([y_train, y_test]).name
            print('target', 'target')
            train_data = pd.concat([X_train, y_train], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)
            col_mappings = {col_name: f'f{col_idx}' 
                            for col_idx, col_name in enumerate(feature_names)}
            train_data.rename(columns=col_mappings, inplace=True)
            test_data.rename(columns=col_mappings, inplace=True)

            TIMER.tic()
            _, _, _, clf = Auto_ViML(train_data, target, test_data)
            training_time = TIMER.tocvalue()

            TIMER.tic()
            y_pred = clf['target_predictions'].astype(int).values.tolist()
            test_time = TIMER.tocvalue()

            collect_and_persist_results(y_test, y_pred, training_time, test_time, "autoviml", SEED)

        except Exception as e:
            print(f'Cannot run autoviml for dataset {get_dataset_ref()} (seed={SEED}). Reason: {str(e)}')
