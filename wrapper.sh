#!/bin/bash

if [ $# -eq 0 ]; then
    >&2 echo "No arguments provided"
    exit 1
fi

while [[ "$#" -gt "0" ]]; do
    case "$1" in
        --preprocess)
          echo "Preprocessing dataset"
          if [[ ! -d "/workspace/data/raw_data" ]]; then
            echo "Raw dataset directory doesn't exist"
            exit 1
          fi
          python start-workflow.py --config-file ${CONFIG_FILE:-"/workspace/configs/workflow-data-preprocessing.yaml"} --mode 1
          shift
          ;;
      --gnn-analytics)
          if [[ ! -f "/DATA_IN/processed_data.csv" ]]; then
            echo "Preprocessed CSV file doesn't exist. Use --preprocess to run the preprocess first"
            exit 1
          fi
          mkdir -p /GNN_TMP
          ./host/script/run_gnn_wf_docker.sh /CONFIGS/workflow-gnn-training.yaml
          shift
          ;;
      --xgb-training)
          if [[ ! -f "/workspace/data/node_edge_data/tabular_with_gnn_emb.csv" ]]; then
            echo "tabular_with_gnn_emb.csv file doesn't exist. Please, use --gnn-workflow to create the file"
            exit 1
          fi
          mkdir -p /workspace/tmp/models
          python start-workflow.py --config-file ${CONFIG_FILE:-"/workspace/configs/workflow-xgb-training.yaml"} --mode 1
          shift
          ;;
      --baseline-training)
          if [[ ! -f "/workspace/data/edge_data/processed_data.csv" ]]; then
            echo "Preprocessed CSV file doesn't exist. Use --preprocess to run the preprocess first"
            exit 1
          fi
          mkdir -p /workspace/tmp/models
          python start-workflow.py --config-file ${CONFIG_FILE:-"/workspace/configs/workflow-baseline.yaml"} --mode 1
          shift
          ;;
      --xgb-serve)
          serve_config=${SERVE_CONFIG:-"/workspace/configs/workflow-xgb-serve.yaml"}
          python fraud_detect/serve/serve.py --config ${serve_config}
          shift
          ;;
      **)
          echo "Wrong argument passed"
          exit 1
    esac
  done
