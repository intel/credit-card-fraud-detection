import os
import yaml
import psutil

def run(processed_data, gnn_tmp, out_dir, csv_dataset, yaml_path):
    settings = {}
    with open(yaml_path, "r+") as yaml_file:
        settings.update(yaml.safe_load(yaml_file))
    work_space = os.getenv("WORKSPACE")
    exec_script = f"{work_space}/src/graphsage_fraud_transductive.py"

    csvdataset=f"{gnn_tmp}/{csv_dataset}"
    if not os.path.exists(csvdataset):
        print(f"{csvdataset} does not exist. Need to build graph")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    model_out=f"{out_dir}/model_graphsage_2L_64.pt"
    nemb_out=f"{out_dir}/node_emb.pt"
    nemb_out_mapped=f"{out_dir}/node_emb_mapped.pt"
    out_data=f"{out_dir}/tabular_with_gnn_emb.csv"
    log_dir=f"{work_space}/logs"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logname=f"log_tabformer_full_homo_1n"
    # minibatch size on each host
    mb_size=settings["workflow_spec"]["dataloader_params"]["batch_size"]
    mb_size_eval=settings["workflow_spec"]["dataloader_params"]["batch_size_eval"]
    # hidden feature size
    hidden_fs=settings["workflow_spec"]["model_params"]["hidden_size"]
    #number of layers in GNN encoder
    n_layers=settings["workflow_spec"]["model_params"]["num_layers"]
    #Learning rate
    l_rate=settings["workflow_spec"]["model_params"]["learning_rate"]
    # fanout per layer
    fan_out=settings["workflow_spec"]["sampler_params"]["fan_out"]
    # num epochs to run for
    epochs=settings["workflow_spec"]["training_params"]["num_epochs"]
    eval_every=settings["workflow_spec"]["training_params"]["eval_every"]
    num_threads=psutil.cpu_count(logical=False)

    cmd_line=f"OMP_NUM_THREADS={num_threads} numactl -N 0 python -u {exec_script} --num_epoch {epochs}  --num_hidden {hidden_fs} --num_layers {n_layers} --lr {l_rate} --fan_out {fan_out} --batch_size {mb_size} --batch_size_eval {mb_size_eval} --eval_every {eval_every} --CSVDataset_dir {csvdataset} --model_out {model_out} --nemb_out {nemb_out} 2>&1 | tee {log_dir}/{logname}.txt"
    os.system(cmd_line)