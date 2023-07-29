#!/bin/bash

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

PROCESSED_DATA=$1
GNN_TMP=$2
OUT_DIR=$3
CSVDATASET=$4
yamlPath=$5

if [ "$#" -ne 5 ]
then
  echo "Incorrect number of arguments to run_train_single.sh"
  exit 1
fi
function parse_yaml {
   local prefix=$6
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|,$s\]$s\$|]|" \
        -e ":1;s|^\($s\)\($w\)$s:$s\[$s\(.*\)$s,$s\(.*\)$s\]|\1\2: [\3]\n\1  - \4|;t1" \
        -e "s|^\($s\)\($w\)$s:$s\[$s\(.*\)$s\]|\1\2:\n\1  - \3|;p" $1 | \
   sed -ne "s|,$s}$s\$|}|" \
        -e ":1;s|^\($s\)-$s{$s\(.*\)$s,$s\($w\)$s:$s\(.*\)$s}|\1- {\2}\n\1  \3: \4|;t1" \
        -e    "s|^\($s\)-$s{$s\(.*\)$s}|\1-\n\1  \2|;p" | \
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)-$s[\"']\(.*\)[\"']$s\$|\1$fs$fs\2|p" \
        -e "s|^\($s\)-$s\(.*\)$s\$|\1$fs$fs\2|p" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p" | \
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]; idx[i]=0}}
      if(length($2)== 0){  vname[indent]= ++idx[indent] };
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) { vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, vname[indent], $3);
      }
   }'
}

eval $(parse_yaml $yamlPath)

EXEC_SCRIPT=${WORKSPACE}/src/graphsage_fraud_transductive.py

#tabformer CSVDataset
CSVDATASET=${GNN_TMP}/${CSVDATASET}
if [[ ! -d ${CSVDATASET} ]]; then
    echo -e "\n${CSVDATASET} does not exist. Need to build graph"
fi;

#model, embb and final csv outputs
mkdir -p ${OUT_DIR}
MODEL_OUT=${OUT_DIR}/model_graphsage_2L_64.pt
NEMB_OUT=${OUT_DIR}/node_emb.pt #these are the default names
NEMB_OUT_MAPPED=${OUT_DIR}/node_emb_mapped.pt #these are the default names
OUT_DATA=${OUT_DIR}/tabular_with_gnn_emb.csv

GRAPH_NAME=( "tabformer_full_homo" )

# Folder and filename where you want your logs.
logdir=${WORKSPACE}/logs
mkdir -p $logdir
logname=log_${GRAPH_NAME}_1n_$RANDOM
echo $logname

cfg="$env_config_path/$env_train_config_file"
eval $(parse_yaml $yamlPath)
# minibatch size on each host
MB_SIZE=$workflow_spec_dataloader_params_batch_size
MB_SIZE_EVAL=$workflow_spec_dataloader_params_batch_size_eval

# hidden feature size
HIDDEN_FS=$workflow_spec_model_params_hidden_size

#number of layers in GNN encoder
N_LAYERS=$workflow_spec_model_params_num_layers

#Learning rate
L_RATE=$workflow_spec_model_params_learning_rate

# fanout per layer
FANOUT=$workflow_spec_sampler_params_fan_out

# num epochs to run for
EPOCHS=$workflow_spec_training_params_num_epochs

EVAL_EVERY=$workflow_spec_training_params_eval_every

#seting number of OMP_NUM_THREADS to number of physical cores in one socket - number of dataloader workers
NUM_CORES_PER_SOCKET=`lscpu | grep "Core(s) per socket" |grep -Eo '[0-9]{1,3}'`
NUM_THREADS=$((NUM_CORES_PER_SOCKET-workflow_spec_dataloader_params_num_workers))
echo -e "\nSetting OMP_NUM_THREADS=$NUM_THREADS"

OMP_NUM_THREADS=$NUM_THREADS numactl -N 0 python -u ${EXEC_SCRIPT} --num_epoch ${EPOCHS}  --num_hidden ${HIDDEN_FS} --num_layers ${N_LAYERS} --lr ${L_RATE} --fan_out ${FANOUT} --batch_size ${MB_SIZE} --batch_size_eval ${MB_SIZE_EVAL} --eval_every ${EVAL_EVERY} --CSVDataset_dir ${CSVDATASET} --model_out ${MODEL_OUT} --nemb_out ${NEMB_OUT} |& tee ${logdir}/${logname}.txt
