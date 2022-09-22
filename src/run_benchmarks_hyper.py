# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
Master code for executing hyperparameter tuning benchmarks
'''

if __name__ == "__main__":

    import argparse
    import logging
    logging.getLogger('matplotlib.font_manager').disabled = True
    import numpy as np
    np.random.seed(42)
    from utils.training import split_data, DBSCAN_Clustering, lgbm_model_hyper
    from utils.data_processing import read_data, filter_clusters
    import joblib
    import pathlib

    parser = argparse.ArgumentParser()

    parser.add_argument('-l',
                        '--logfile',
                        type=str,
                        default="",
                        help="log file to output benchmarking results to")

    parser.add_argument('-i',
                        '--intel',
                        default=False,
                        action="store_true",
                        help="use intel accelerated technologies where available")

    FLAGS = parser.parse_args()

    if FLAGS.logfile == "":
        logging.basicConfig(level=logging.DEBUG)
    else:
        path = pathlib.Path(FLAGS.logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=FLAGS.logfile, level=logging.DEBUG)
        
    logger = logging.getLogger()
    intel_flag = FLAGS.intel

    MODEL_FILE1 = 'LGBM_Classifier.pkl'

    filepath = "./data/creditcard.csv"
    
    logger.info("Reading Data...")
    credit_card_data = read_data(filepath)
    logger.info("Splitting Data into Train/Test...")
    X_train, X_test, y_train, y_test = split_data(credit_card_data)

    most_important_names = ['V16', 'V14']
    eps_val = 0.3
    minimum_samples = 20
    logger.info("=======> Running DBSCAN Clustering and filtering for classification...")
    df_clusters, cluster_time = DBSCAN_Clustering(X_train, most_important_names, eps_val, minimum_samples, intel_flag)
    X_train_clustered, y_train_clustered = filter_clusters(X_train, y_train, df_clusters)
    logger.info("=======> DBSCAN Clustering time = %s", str(cluster_time))
    logger.info("\n")

    param_grid = {'max_depth': [5, 10], 'min_child_weight': [3], 'n_estimators': [500, 750],
                  'num_leaves': [5], 'reg_alpha': [0.5], 'reg_lambda': [0.5], 'metric': ['auc'],
                  'boosting_type': ['gbdt'], 'colsample_bytree': [.8], 'subsample': [.9],
                  'min_split_gain': [.01], 'max_bin': [20, 25], 'learning_rate': [0.01]}

    logger.info("=======> Length of training dataframe post clustering = %s", str(len(X_train_clustered)))
    logger.info("=======> Value counts of labeled data on training set = %s", str(sum(y_train_clustered)/len(y_train_clustered)))
    logger.info("=======> Value counts of labeled data on test set = %s", str(sum(y_test)/len(y_test)))
    logger.info("=======> Length of full training dataframe = %s", str(len(X_train)))
    logger.info("\n")

    logger.info("=======> Training model_cluster on clustered data")
    lgb_model, train_time = lgbm_model_hyper(X_train_clustered, y_train_clustered, param_grid)
    logger.info("=======> LGBM training time for clustered data = %s", str(train_time))
    MODEL_FILE = 'Clustered_' + MODEL_FILE1
    joblib.dump(lgb_model, MODEL_FILE)
    
    logger.info("\n")

    logger.info("=======> Training model on Full data")
    lgb_model_full, train_time = lgbm_model_hyper(X_train, y_train, param_grid)
    logger.info("=======> LGBM training time for full data = %s", str(train_time))
    MODEL_FILE = 'Full_' + MODEL_FILE1
    joblib.dump(lgb_model_full, MODEL_FILE)
