from pycaret.classification import *

from common import *

if __name__ == "__main__":

    for SEED in PRIME_NUMBERS:
        
        try:

            set_random_seed(SEED)
                
            X_train, X_test, y_train, y_test = load_data_delegate(SEED)
            train_df = pd.DataFrame(X_train).assign(**{'class': pd.Series(y_train)}).dropna()
            test_df = pd.DataFrame(X_test).assign(**{'class': pd.Series(y_test)}).dropna()

            clf = setup(
                data=train_df,
                target='class', 
                session_id=SEED,
                log_experiment=True, 
                experiment_name=f'automl_pycaret_{get_dataset_ref()}_{SEED}',
                test_data=test_df,
                fold=5,
                n_jobs=NUM_CPUS
            )

            TIMER.tic()
            best_model = clf.compare_models(budget_time=EXEC_TIME_MINUTES, n_select=1, sort='Accuracy')
            training_time = TIMER.tocvalue()

            TIMER.tic()
            y_pred = best_model.predict(X_test)
            # y_pred = predict_model(best_model, data=X_test)['prediction_label'].values
            # print('________ y_pred = ', y_pred)
            test_time = TIMER.tocvalue()

            collect_and_persist_results(y_test, y_pred, training_time, test_time, "pycaret", SEED)

        except Exception as e:
            print(f'Cannot run pycaret for dataset {get_dataset_ref()} (seed={SEED}). Reason: {str(e)}')
