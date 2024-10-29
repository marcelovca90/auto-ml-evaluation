    # Attempts

    # # AutoViML
    # echo ======== AutoViML ========
    # python3.8 -m venv venv-autoviml
    # source ./venv-autoviml/bin/activate
    # python3.8 -m pip install --upgrade pip
    # python3.8 -m pip install --upgrade setuptools pytictoc wheel pandas scikit-learn scikit-multilearn
    # python3.8 -m pip install git+https://github.com/AutoViML/Auto_ViML.git@ef136a439eaa275b345868066a127aa2d019542d
    # sed -i "s/get_ipython/#get_ipython/" venv-autoviml/lib/python3.8/site-packages/autoviml/QuickML_Ensembling.py
    # sed -i "s/os.cpu_count()/1/" venv-autoviml/lib/python3.8/site-packages/autoviml/Auto_ViML.py
    # sed -i "s/n_jobs = -1/n_jobs = 1/" venv-autoviml/lib/python3.8/site-packages/autoviml/Auto_ViML.py
    # sed -i "s/n_jobs=-1/n_jobs=1/" venv-autoviml/lib/python3.8/site-packages/autoviml/Auto_ViML.py
    # sed -i "s/nthread=-1/nthread=1/" venv-autoviml/lib/python3.8/site-packages/autoviml/Auto_ViML.py
    # sed -i "s/\['nthread'\] = -1/['nthread'] = 1/" venv-autoviml/lib/python3.8/site-packages/autoviml/Auto_ViML.py
    # sed -i "s/train = part_train.append(part_cv)/train = pd.concat([part_train, part_cv], axis=0)/" venv-autoviml/lib/python3.8/site-packages/autoviml/Auto_ViML.py
    # python3.8 ./automl_autoviml.py $id
    # pkill -f viml
    # sleep 10

    # # BlobCity AutoAI
    # echo ======== BlobCity AutoAI ========
    # python3.8 -m venv venv-blob
    # source ./venv-blob/bin/activate
    # python3.8 -m pip install --upgrade pip
    # python3.8 -m pip install --upgrade setuptools pytictoc wheel Cython "typing-extensions<4.6.0,>=3.6.6"
    # python3.8 -m pip install git+https://github.com/keras-team/keras-tuner.git
    # python3.8 -m pip install scikit-learn scikit-multilearn opencv-python statsmodels IPython autokeras blobcity
    # sed -i 's/"auto",//' venv-blob/lib/python3.8/site-packages/blobcity/config/classifier_config.py
    # sed -i 's/"auto",//' venv-blob/lib/python3.8/site-packages/blobcity/config/regressor_config.py
    # python3.8 ./automl_blob.py $id
    # pkill -f blob
    # sleep 10