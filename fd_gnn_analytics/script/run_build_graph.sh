#!/bin/bash

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

data_prePath=$1
tmpPath=$2
config=$3
CSVDataset_name=$4

#build graph (will create CSVDataset folder and save csv files)
python ${WORKSPACE}/src/build_graph.py --data_in ${data_prePath} --gnn_tmp ${tmpPath} --tab2graph_cfg ${config} --CSVDataset_name ${CSVDataset_name}
