#!/bin/bash
BASEDIR=$(dirname "$0")
if [ $# -ne 6 ]; then
    echo "No arguments provided"
    exit 1
fi

if [[ "$1" != "--config-path" ]]; then
    echo "No 'config-path' parameter provided"
    exit 1
fi

if [[ "$3" != "--input-path" ]]; then
    echo "No 'input-path' parameter provided"
    exit 1
fi
if [[ "$5" != "--output-path" ]]; then
    echo "No 'output-path' parameter provided"
    exit 1
fi

export CONFIG_PATH=$2
export DATA_IN_FILE=$4/processed_data.csv
export DATA_OUT=$6
mkdir -p ${DATA_OUT}

export WORKSPACE=${BASEDIR}

mkdir -p /GNN_TMP
bash ${BASEDIR}/script/run_gnn_wf_docker.sh ${BASEDIR}/config/config.yaml