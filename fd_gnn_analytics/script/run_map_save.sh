#!/bin/bash

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

PROCESSED_DATA=$1
OUT_DIR=$2
CONFIG=$3
OUT_DATA=${OUT_DIR}/tabular_with_gnn_emb.csv

if [[ ! -f "${PROCESSED_DATA}" ]]; then
    echo -e "\n${PROCESSED_DATA} does not exist and is needed to map node embeddings"
fi

python ${WORKSPACE}/src/map_emb_single.py --processed_data_path ${PROCESSED_DATA} --model_emb_path ${OUT_DIR} --out_data ${OUT_DATA} --tab2graph_cfg ${CONFIG}
