# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

from queue import Empty
import pandas as pd
import numpy as np
import csv

import time
import yaml
import os
import argparse
import dgl
from collections import OrderedDict


def main(args):
    with open(args.tab2graph_cfg, "r") as file:
        config = yaml.safe_load(file)

    CSVDataset_dir = os.path.join(args.gnn_tmp, args.CSVDataset_name)
    print(CSVDataset_dir)
    for dir in [args.gnn_tmp, CSVDataset_dir]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    # 1.   load CSV file output of Classical ML edge featurization workflow
    print("loading processed data")
    start = time.time()
    df = pd.read_csv(args.data_in)  # , nrows=10000)
    t_load_data = time.time()
    print("time lo load processed data", t_load_data - start)

    # 2.   Renumbering - generating node/edge ids starting from zero
    print("Node renumbering")

    def column_index(series, offset=0):
        return {k: v + offset for v, k in enumerate(series.value_counts().index.values)}

    # create dictionary of dictionary to stare node mapping for all node types
    offset = 0
    dict = OrderedDict()
    # create mapping dictionary between original IDs and incremental IDs starting at zero
    col_map = {}
    for i, node in enumerate(config["node_columns"]):
        key = str(node + "_2idx")
        dict[key] = column_index(df[config["node_columns"][i]], offset=offset)
        new_col_name = node + "_Idx"
        col_map[node] = new_col_name
        # add new Idx to dataframe
        df[new_col_name] = df[config["node_columns"][i]].map(dict[key])
        # offset = len(dict[key]) #remove if doing hetero mappint where all types start from zero
    t_renum = time.time()
    print("re-enumerated column map: ", col_map)
    print("time to renumerate", t_renum - t_load_data)

    # 3    create masks for train, val and test splits (add new columns with masks)
    if config["edge_split"]:
        df = pd.concat(
            [
                df,
                pd.get_dummies(
                    df[config["edge_split"]].astype("category"), prefix="masks"
                ),
            ],
            axis=1,
        )

    # 4    Prepare CSVDataset files for DGL to ingest and create graph
    print("Writting data into set of CSV files (nodes/edges)")
    # The specs for yaml file content and node/edge CSV file please refer to:
    # https://docs.dgl.ai/en/1.0.x/guide/data-loadcsv.html#guide-data-pipeline-loadcsv

    # programatically create meta.yaml expected by DGL from yaml config
    list_of_n_dict = []
    for i, node in enumerate(col_map.keys()):
        list_of_n_dict.append(
            {"file_name": "nodes_" + str(i) + ".csv", "ntype": col_map[node]}
        )

    list_of_e_dict = []
    for i, edge_type in enumerate(config["edge_types"]):
        # replace node ids with the reenumerated Idx
        edge_type[0] = col_map[edge_type[0]]
        edge_type[2] = col_map[edge_type[2]]
        print(edge_type)
        # keep edge type from tabular2graph.yaml and update node type to the new Idx ones
        list_of_e_dict.append(
            {"file_name": "edges_" + str(i) + ".csv", "etype": edge_type}
        )

    with open(os.path.join(CSVDataset_dir, "meta.yaml"), "w") as f:
        python_data = {
            "dataset_name": args.CSVDataset_name,
            "node_data": list_of_n_dict,
            "edge_data": list_of_e_dict,
        }
        data = yaml.dump(python_data, f, sort_keys=False, default_flow_style=False)

    with open(os.path.join(CSVDataset_dir, "meta.yaml"), "r") as file:
        meta_yaml = yaml.safe_load(file)
    print("\nmeta_yaml: \n", meta_yaml)

    # DGL node/edge csv headers
    # edge_header = ["src_id", "dst_id", "label", "train_mask", "val_mask","test_mask","feat"]
    # node_header = ["node_id", "label", "train_mask", "val_mask","test_mask","feat"]

    # write edges_.csv files
    for i, edge_type in enumerate(meta_yaml["edge_data"]):
        print("\nWriting: ", edge_type["file_name"])
        edge_header = ["src_id", "dst_id"]  # minimum required
        edge_df_cols = [edge_type["etype"][0], edge_type["etype"][2]]
        if config["edge_label"]:
            edge_header.append("label")
            edge_df_cols.append(config["edge_label"])
        if config["edge_split"]:
            # it is required that split has 3 values (0,1,2) for train test val respectively
            edge_header.extend(["train_mask", "val_mask", "test_mask"])
            edge_df_cols.extend(["masks_0", "masks_1", "masks_2"])
            # edge_header.extend(["train_mask"])
            # edge_df_cols.extend(["masks_0"])
        if config["edge_features"]:
            feat_keys = config["edge_features"]
            print("features for CSVDataset edges: ", feat_keys)
            # Note: feat_as_str needs to be a string of comma separated values enclosed in double quotes for dgl default parser to work
            df["edge_feat_as_str"] = df[feat_keys].astype(str).apply(",".join, axis=1)
            edge_header.append("feat")
            edge_df_cols.append("edge_feat_as_str")
        assert len(edge_df_cols) == len(edge_header)
        df[edge_df_cols].to_csv(
            os.path.join(CSVDataset_dir, edge_type["file_name"]),
            index=False,
            header=edge_header,
        )
    # write nodes_.csv files
    for i, node in enumerate(meta_yaml["node_data"]):
        print("\nWriting: ", meta_yaml["node_data"][i]["file_name"])
        print(df[meta_yaml["node_data"][i]["ntype"]].unique())
        np.savetxt(
            os.path.join(CSVDataset_dir, meta_yaml["node_data"][i]["file_name"]),
            df[meta_yaml["node_data"][i]["ntype"]].unique(),
            delimiter=",",
            header="node_id",
            comments="",
        )

    t_csvDataset = time.time()
    print("time to write CSVDatasets", t_csvDataset - t_renum)

    # for testing purposes
    # dataset = dgl.data.CSVDataset(CSVDataset_dir, force_reload=True)
    # print(dataset[0])


def file(raw_path):
    if not os.path.isfile(raw_path):
        raise argparse.ArgumentTypeError(
            '"{}" is not an existing file'.format(raw_path)
        )
    return os.path.abspath(raw_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BuildGraph")
    parser.add_argument(
        "--data_in",
        type=file,
        default="/DATA_IN/processed_data.csv",
        help="Input file with path (processed_data.csv) ",
    )
    parser.add_argument(
        "--gnn_tmp", default="/GNN_TMP/", help="The path to the gnn_tmp"
    )
    parser.add_argument(
        "--tab2graph_cfg", required=True, help="The path to the tabular2graph.yaml"
    )
    parser.add_argument("--CSVDataset_name", type=str, help="CSVDataset name")

    args = parser.parse_args()

    print(args)
    main(args)
