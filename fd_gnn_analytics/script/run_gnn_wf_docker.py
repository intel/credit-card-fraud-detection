import os
import yaml
from script import run_build_graph, run_train_single, run_map_save

def run(config_path):
    settings = {}
    with open(config_path, "r+") as config_file:
        settings.update(yaml.safe_load(config_file))

    print("Starting single node workflow...")
    if settings["single"]["build_graph"]:
        print("Building graph...")
        graph_config = os.path.join(os.getenv("CONFIG_PATH"),"tabular2graph.yaml")
        data_in_file = os.getenv("DATA_IN_FILE")
        graph_CSVDataset_name = settings["graph"]["CSVDataset_name"]
        run_build_graph.run(data_in_file, "/GNN_TMP", graph_config, graph_CSVDataset_name)
    if settings["single"]["gnn_training"]:
        print("Start GNN training...")
        graph_config = os.getenv("CONFIG_PATH")
        graph_config=f"{graph_config}/gnn-training.yaml"
        data_in_file = os.getenv("DATA_IN_FILE")
        data_out = os.getenv("DATA_OUT")
        graph_CSVDataset_name = settings["graph"]["CSVDataset_name"]
        run_train_single.run(data_in_file, "/GNN_TMP", data_out, graph_CSVDataset_name, graph_config)
    if settings["single"]["map_save"]:
        print("Mapping to original graph IDs followed by mapping to CSV file output")
        print("This may take a while")
        graph_config = os.getenv("CONFIG_PATH")
        graph_config = f"{graph_config}/tabular2graph.yaml"
        data_in_file = os.getenv("DATA_IN_FILE")
        data_out = os.getenv("DATA_OUT")
        run_map_save.run(data_in_file, data_out, graph_config)