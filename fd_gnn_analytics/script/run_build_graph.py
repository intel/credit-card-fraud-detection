import os

def run(data_prePath, tmpPath, config, CSVDataset_name):
    workspace = os.getenv("WORKSPACE")
    cmd_line = f"python {workspace}/src/build_graph.py --data_in {data_prePath} --gnn_tmp {tmpPath} --tab2graph_cfg {config} --CSVDataset_name {CSVDataset_name}"
    os.system(cmd_line)