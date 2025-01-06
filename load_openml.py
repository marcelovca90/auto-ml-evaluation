def load_openml(seed):
    dataset = fetch_openml(data_id=get_dataset_ref(), return_X_y=False)
    X, y = dataset.data.copy(deep=True), dataset.target.copy(deep=True)

    # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9321731
    # https://github.com/mrapp-ke/Boomer-Datasets/raw/refs/heads/main/flags.arff
    if get_dataset_ref() == 285: # flags
        df = pd.concat([X, y], axis='columns')
        label_columns = [
            'crescent', 'triangle', 'icon', 'animate', 'text', 'red',
            'green', 'blue', 'gold', 'white', 'black', 'orange'
        ]
        y = df[label_columns].astype(int)  # Select only label columns
        for col in y.columns.values:
            y[col] = y[col].map({0: 'FALSE', 1: 'TRUE'})
        X = df.drop(columns=label_columns).infer_objects()  # Drop label columns to get remaining ones
        for col in X.columns:
            if col not in ['mainhue', 'topleft', 'botright']:
                X[col] = X[col].astype(float)
        assert df.shape[0] == X.shape[0] # rows
        assert df.shape[0] == y.shape[0] # rows
        assert df.shape[1] == X.shape[1] + y.shape[1] # columns

    # handle categorical features
    for col in X.columns.values:
        if X[col].dtype.name == 'category':
            X.loc[:, col] = pd.Series(pd.factorize(X[col])[0])
    print(X.dtypes)

    if is_multi_label():
        for col in y.columns.values:
            y[col] = y[col].map({'FALSE': 0, 'TRUE': 1}).to_numpy()
        if LABEL_POWERSET:
            y = pd.Series(LabelPowerset().transform(y))
            print(y.unique())
        else:
            print(y.columns)
    else:
        y = pd.Series(pd.factorize(y)[0])
        print(y.unique())
    return train_test_split(X, y, test_size=0.2, random_state=seed)