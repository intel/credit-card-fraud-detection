#!/bin/bash

if [ $# -ne 6 ]; then
    echo "No arguments provided"
    exit 1
fi

if [[ "$1" != "--config-path" ]]; then
    echo "No 'config-path' parameter provided"
    exit 1
fi

if [[ "$3" != "--input-file" ]]; then
    echo "No 'input-file' parameter provided"
    exit 1
fi
if [[ "$5" != "--output-path" ]]; then
    echo "No 'output-path' parameter provided"
    exit 1
fi

export CONFIG_PATH=$2
export DATA_IN_FILE=$4
export DATA_OUT=$6
mkdir -p ${DATA_OUT}

export WORKSPACE=/cnvrg/fd_gnn_analytics

mkdir -p /GNN_TMP
bash /cnvrg/fd_gnn_analytics/script/run_gnn_wf_docker.sh ${CONFIG_PATH}/config.yaml