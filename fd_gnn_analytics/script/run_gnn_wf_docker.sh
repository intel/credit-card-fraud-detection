#!/bin/bash

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

yamlPath="$1"

function parse_yaml {
   local prefix=$2
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

echo -e "\nStarting single node workflow..."
if [ "$single_build_graph" = True ]; then
    echo -e "\nBuilding graph..."
    graph_config="${CONFIG_PATH}/tabular2graph.yaml"
    bash ${WORKSPACE}/script/run_build_graph.sh "${DATA_IN_FILE}" /GNN_TMP ${graph_config} ${graph_CSVDataset_name}
fi;
if [ "${single_gnn_training}" = True ]; then
    echo -e "\nStart GNN training..."
    graph_config="${CONFIG_PATH}/gnn-training.yaml"
    echo $graph_config
    bash ${WORKSPACE}/script/run_train_single.sh "${DATA_IN_FILE}" /GNN_TMP "${DATA_OUT}" ${graph_CSVDataset_name} "${graph_config}"
fi;
if [ "${single_map_save}" = True ]; then
    echo "Mapping to original graph IDs followed by mapping to CSV file output"
    echo "This may take a while"
    graph_config="${CONFIG_PATH}/tabular2graph.yaml"
    bash ${WORKSPACE}/script/run_map_save.sh "${DATA_IN_FILE}" "${DATA_OUT}" "${graph_config}"
fi;
