# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF

import dgl
import dgl.nn as dglnn
from dgl.nn import SAGEConv
from dgl.dataloading import (
    DataLoader,
    NeighborSampler,
    MultiLayerFullNeighborSampler,
    as_edge_prediction_sampler,
    negative_sampler,
)
from dgl import save_graphs, load_graphs
from dgl import DGLGraph

import tqdm
import argparse
import time
from sklearn.metrics import roc_auc_score
import numpy as np
import os


def main(args):
    # add random seed
    print("random seed used in partitioning")
    dgl.random.seed(1)

    # create directories to save the partitions
    # partition dir with subfolders for each set of partitions (i,e  tabformer_2parts, tabformer_4parts...)
    part_dir = args.partition_out
    curr_part_dir = os.path.join(part_dir, "tabformer_" + str(args.num_part) + "parts")
    for dir in [part_dir, curr_part_dir]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    # load and preprocess Tabformer dataset
    print("Loading data")
    start = time.time()
    # set force_reload=False if no changes on input graph (much faster otherwise ingestion ~30min)
    dataset = dgl.data.CSVDataset(args.CSVDataset, force_reload=False)
    print("time to load tabformer from CSVs: ", time.time() - start)

    hg = dataset[0]  # only one graph
    print(hg)
    print("etype to read train/test/val from: ", hg.canonical_etypes[0][1])

    E = hg.num_edges(hg.canonical_etypes[0][1])
    reverse_eids = torch.cat([torch.arange(E, 2 * E), torch.arange(0, E)])
    print("First reverse id is:  ", reverse_eids[0])

    # convert graph to homogeneous
    g = dgl.to_homogeneous(hg)
    print(g)

    # part_method='random' works
    # part_method='metis' is giving a "free(): corrupted unsorted chunks error" with multigraph (multiple links between same pair of nodes)
    # part_method='metis' works if you first do "dgl.to_simple(hg)" to keep single edge between pair of noes but that is not appropriate since we want to keep all multigraph edges

    nmap, emap = dgl.distributed.partition_graph(
        g,
        args.graph_name,
        args.num_part,
        num_hops=args.num_hops,
        part_method="random",
        out_path=curr_part_dir,
        balance_edges=True,
        return_mapping=True,
    )

    torch.save(nmap, os.path.join(curr_part_dir, "nmap.pt"))
    torch.save(emap, os.path.join(curr_part_dir, "emap.pt"))

    # #load first partition to verify
    (
        g,
        node_feats,
        edge_feats,
        gpb,
        graph_name,
        ntypes_list,
        etypes_list,
    ) = dgl.distributed.load_partition(
        os.path.join(curr_part_dir, args.graph_name + ".json"), 0
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphPartition")
    parser.add_argument(
        "--CSVDataset",
        help="Path to CSVDataset graph folder ",
    )
    parser.add_argument(
        "--partition_out",
        help="Folder to store partitions. It will create a subfolder to this for different partition sets",
    )
    parser.add_argument(
        "--graph_name",
        type=str,
        default="tabformer_full_homo",
        help="The path to the partition config file",
    )
    parser.add_argument("--num_part", type=int, default=2, help="number of partitions")
    parser.add_argument(
        "--num_hops",
        type=int,
        default=1,
        help="number of hops of nodes we include in a partition as HALO nodes",
    )

    args = parser.parse_args()

    print(args)
    main(args)
