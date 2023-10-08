import os

def run(processed_data, out_dir, config):
    out_data=f"{out_dir}/tabular_with_gnn_emb.csv"
    if not os.path.exists(processed_data):
        print(f"{processed_data} does not exist and is needed to map node embeddings")
    work_space = os.getenv("WORKSPACE")
    cmd_line = f"python {work_space}/src/map_emb_single.py --processed_data_path {processed_data} --model_emb_path {out_dir} --out_data {out_data} --tab2graph_cfg {config}"
    os.system(cmd_line)