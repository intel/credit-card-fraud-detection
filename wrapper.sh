#!/bin/bash

if [ $# -eq 0 ]; then
    >&2 echo "No arguments provided"
    exit 1
fi

while [[ "$#" -gt "0" ]]; do
    case "$1" in
        --preprocess)
          echo "Preprocessing dataset"
          if [[ ! -d "/fraud-detection/data/raw_data" ]]; then
            echo "Raw dataset directory doesn't exist"
            exit 1
          fi
          python start-workflow.py --config-file ${CONFIG_FILE:-"/workspace/configs/workflow-data-preprocessing.yaml"}
          shift
          ;;
      --gnn-analytics)
          if [[ ! -f "/fraud-detection/data/edge_data/processed_data.csv" ]]; then
            echo "Preprocessed CSV file doesn't exist. Use --preprocess to run the preprocess first"
            exit 1
          fi
          mkdir -p /fraud-detection/data/node_edge_data
          mkdir -p /fraud-detection/checkpoint
          python gnn_workflow.py --data_in /fraud-detection/data/edge_data/processed_data.csv --data_out /fraud-detection/data/node_edge_data/tabformer_with_gnn_emb.csv --gnn_tmp /fraud-detection/checkpoint
          shift
          ;;
      --xgb-training)
          if [[ ! -f "/fraud-detection/data/node_edge_data/tabformer_with_gnn_emb.csv" ]]; then
            echo "tabformer_with_gnn_emb.csv file doesn't exist. Please, use --gnn-workflow to create the file"
            exit 1
          fi
          mkdir -p /workspace/tmp/models
          python start-workflow.py --config-file ${CONFIG_FILE:-"/workspace/configs/workflow-xgb-training.yaml"}
          shift
          ;;
      --baseline-training)
          if [[ ! -f "/fraud-detection/data/edge_data/processed_data.csv" ]]; then
            echo "Preprocessed CSV file doesn't exist. Use --preprocess to run the preprocess first"
            exit 1
          fi
          mkdir -p /workspace/tmp/models
          python start-workflow.py --config-file ${CONFIG_FILE:-"/workspace/configs/workflow-baseline.yaml"}
          shift
          ;;
        **)
          echo "Wrong argument passed"
          exit 1
    esac
  done
