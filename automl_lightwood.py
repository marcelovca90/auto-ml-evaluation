import pandas as pd
import numpy as np
from lightwood.api.high_level import (
    ProblemDefinition, json_ai_from_problem,
    code_from_json_ai, predictor_from_code)

from common import *

if __name__ == "__main__":

    for SEED in PRIME_NUMBERS:
        
        try:

            set_random_seed(SEED)

            X_train, X_test, y_train, y_test = load_data_delegate(SEED)
            train_df = pd.DataFrame(X_train).assign(**{'class': pd.Series(y_train)}).dropna()
            test_df = pd.DataFrame(X_test).assign(**{'class': pd.Series(y_test)}).dropna()

            problem_def = ProblemDefinition.from_dict({
                'target': 'class', 'time_aim': EXEC_TIME_SECONDS, 
                'seed_nr': SEED, 'strict_mode': False}
            )

            TIMER.tic()
            json_ai = json_ai_from_problem(train_df, problem_definition=problem_def)
            code = code_from_json_ai(json_ai)
            clf = predictor_from_code(code)
            clf.learn(train_df)
            training_time = TIMER.tocvalue()

            TIMER.tic()
            y_pred = clf.predict(test_df).prediction.astype(y_test.dtype)
            test_time = TIMER.tocvalue()

            collect_and_persist_results(y_test, y_pred, training_time, test_time, "lightwood", SEED)

        except Exception as e:
            print(f'Cannot run lightwood for dataset {get_dataset_ref()} (seed={SEED}). Reason: {str(e)}')
