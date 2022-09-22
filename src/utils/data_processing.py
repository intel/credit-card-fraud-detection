# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
This code has the functions needed for reading and processing data
'''

import pandas as pd

def read_data(data_path):
    credit_card_data = pd.read_csv(data_path)
    return credit_card_data

def filter_clusters(X_train, y_train, df_clusters):
    df_clusters['Class'] = y_train
    class_mask = df_clusters['Class'] == 1
    class_fitered_df = df_clusters[class_mask]
    relevant_clusters = class_fitered_df['Clusters'].value_counts().index.values[0]  # select the top two
    
    print("Selecting following clusters which has most positive classes: ", relevant_clusters)
    
    X_train['Clusters'] = df_clusters['Clusters']
    mask = X_train['Clusters'] == relevant_clusters
    X_train_clustered = X_train[mask]
    y_train_clustered = y_train[mask]

    return X_train_clustered, y_train_clustered
