# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
This code has the functions needed for train-test-split, clustering and model training
'''

import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

def split_data(raw_data):
    features_data = raw_data.drop(columns=['Class'])
    class_data = raw_data['Class']
    X_train, X_test, y_train, y_test = train_test_split(features_data, class_data, stratify=class_data, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def DBSCAN_Clustering(data_raw, features_of_interest, epsilon, min_samp, flag):
    if flag:
        from sklearnex import patch_sklearn  # pylint: disable=E0401, C0415
        patch_sklearn()
    from sklearn.cluster import DBSCAN  # pylint: disable=C0415
    scaler = StandardScaler()
    data_for_clustering = data_raw[features_of_interest]
    data_for_clustering_scaled = scaler.fit_transform(data_for_clustering)
    lst_clustering_time = []
    for _i in [1, 2, 3, 4, 5]:
        start_time = time.time()
        db = DBSCAN(eps=epsilon, min_samples=min_samp, n_jobs=-1).fit(data_for_clustering_scaled)
        lst_clustering_time.append(time.time()-start_time)
    clustering_time = min(lst_clustering_time)
    data_for_clustering['Clusters'] = db.labels_
    return data_for_clustering, clustering_time

def lgbm_model_train(df_for_training, class_for_training, param_dict):
    lgb_classifier = LGBMClassifier(**param_dict, random_state=42)
    lst_training_time = []
    for _i in [1, 2, 3, 4, 5]:
        start_time = time.time()
        lgb_classifier.fit(df_for_training.drop(columns=['Clusters']), class_for_training)
        lst_training_time.append(time.time()-start_time)
    train_time = min(lst_training_time)
    return lgb_classifier, train_time

def lgbm_model_hyper(df_for_training, class_for_training, param_dict):
    lgb_classifier = LGBMClassifier(random_state=42)
    gs = GridSearchCV(lgb_classifier, param_grid=param_dict, cv=3, verbose=3)
    lst_hyper_time = []
    for _i in [1, 2, 3, 4, 5]:
        start_time = time.time()
        gs_results = gs.fit(df_for_training.drop(columns=['Clusters']), class_for_training)
        lst_hyper_time.append(time.time()-start_time)
    hyper_time = min(lst_hyper_time)
    best_grid = gs_results.best_estimator_
    return best_grid, hyper_time
