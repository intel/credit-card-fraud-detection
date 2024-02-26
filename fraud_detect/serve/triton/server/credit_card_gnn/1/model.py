import io
import os
import json

import torch
import torch.nn.functional as F
import dgl
import yaml
import pickle
import numpy as np
import pandas as pd

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils
from category_encoders import TargetEncoder
from dgl.dataloading import MultiLayerFullNeighborSampler, DataLoader

from gnn_model import GNNServeModel


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(model_config, "output__0")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

        # Load Config file
        config_file = os.path.join(args['model_repository'], "1/GNN-serve.yaml")
        self.configs = yaml.safe_load(open(config_file, "r"))
        self.model = GNNServeModel(self.configs["model"]["vocab_size"], self.configs["model"]["hid_size"], self.configs["model"]["n_layers"])
        self.model.load_state_dict(torch.load(os.path.join(args['model_repository'],'1/model_graphsage_2L_64.pt')))
        self.origin_df = pd.read_csv(self.configs["origin_transaction_file"])

    def preprocess(self, data) -> pd.DataFrame:
        df = pd.DataFrame(data, columns=self.origin_df.columns[:-1])
        df = pd.concat([self.origin_df, df], ignore_index=True)
        df.columns = df.columns.str.replace(" ", "_").str.lower()
        # Step 2 : Categorical, one-hot, multi-hot encoding (independent of data split)
        df["card_id"] = df["user"].astype("str") + df["card"].astype("str")
        df["card_id"] = df["card_id"].astype("float32")

        df["merchant_id"] = df["merchant_name"].astype("category").cat.codes
        df["amount"] = df["amount"].str.strip("$").astype("float32")

        df["merchant_city"] = df["merchant_city"].astype("category")
        df["merchant_state"] = df["merchant_state"].astype("category")
        df["zip"] = df["zip"].astype("str").astype("category")
        df["mcc"] = df["mcc"].astype("category")
        df["is_fraud?"] = df["is_fraud?"].fillna(-1)
        df.loc[df["is_fraud?"]!=-1,"is_fraud?"] = df.loc[df["is_fraud?"]!=-1,"is_fraud?"].astype("category").cat.codes
        df["year"] = df["year"].astype("int32")
        df["month"] = df["month"].astype("int32")
        df["day"] = df["day"].astype("int32")

        # One hot encoding `use_chip`
        oneh_enc_cols = ["use_chip"]
        df = pd.concat([df, pd.get_dummies(df[oneh_enc_cols])], axis=1)
        df.drop(
            columns=["use_chip"], axis=1, inplace=True
        )  # we dont need it after being encoded

        # Multi hot encoding the errors
        exploded = df["errors?"].str.strip(",").str.split(",").explode()
        raw_one_hot = pd.get_dummies(exploded, columns=["errors?"])
        errs = raw_one_hot.groupby(raw_one_hot.index).sum()
        df = pd.concat([df, errs], axis=1)
        df.drop(
            columns=["errors?"], axis=1, inplace=True
        )  # we dont need it after being encoded

        # time encoding
        df["time"] = (
            df["time"]
            .str.split(":")
            .apply(lambda x: int(x[0]) * 60 + int(x[1]))
            .astype("uint8")
        )
        df["time"] = (df["time"] - df["time"].min()) / (df["time"].max() - df["time"].min())

        # Step 3 : Data Splitting for target encoding
        df["split"] = pd.Series(np.zeros(df["year"].size), dtype=np.int8)
        df.loc[df["year"] == 2018, "split"] = 1
        df.loc[df["year"] > 2018, "split"] = 2

        # Keep card_id and merchant_id in the validation and test datasets
        # only if they are included in the train datasets.
        train_card_ids = df.loc[df["split"] == 0, "card_id"]
        train_merch_ids = df.loc[df["split"] == 0, "merchant_id"]

        df.loc[(df["split"] != 0) & ~df["card_id"].isin(train_card_ids), "split"] = 3
        df.loc[(df["split"] != 0) & ~df["merchant_id"].isin(train_merch_ids), "split"] = 3
        df.loc[df["is_fraud?"] == -1, "split"] = 2

        # step 4 : Target and label encoding using data splits to avoid information leakage
        train_df = df.loc[df["split"] == 0]
        valtest_df = df.loc[(df["split"] == 1) | (df["split"] == 2)]

        # Target encoding
        high_card_cols = ["merchant_city", "merchant_state", "zip", "mcc"]
        for col in high_card_cols:
            tgt_encoder = TargetEncoder(smoothing=0.001)
            train_df[col] = tgt_encoder.fit_transform(
                train_df[col], train_df["is_fraud?"]
            ).astype("float32")
            # display(train_df[col])
            valtest_df[col] = tgt_encoder.transform(valtest_df[col]).astype("float32")
        return valtest_df.loc[valtest_df["is_fraud?"]==-1]        

    def build_new_graph(self, df: pd.DataFrame):
        with open(self.configs["graph_config_file"], "r") as file:
            config = yaml.safe_load(file)

        # reading the node2ID file
        with open(self.configs["node2ID_file"], "rb") as f:
            node2ID = pickle.load(f)

        # Renumbering - generating node/edge ids
        def column_index(series, node2ID):
            for v, k in enumerate(series.value_counts().index.values):
                if k not in node2ID:
                    node2ID[k] = len(node2ID)
                else:
                    node2ID[k] = node2ID[k]
            return node2ID

        col_map = {}
        offset = 0
        for i, node in enumerate(config["node_columns"]):
            key = str(node + "_2idx")
            node2ID[key] = column_index(df[config["node_columns"][i]], node2ID[key])
            offset = len(node2ID[key]) if i == 0 else 0
            new_col_name = node + "_Idx"
            col_map[node] = new_col_name
            # add new Idx to dataframe
            df[new_col_name] = df[config["node_columns"][i]].map(node2ID[key])
        df["edge_feat_as_str"] = df[config["edge_features"]].astype(str).apply(",".join, axis=1)
        dataframe_keys = [] + list(col_map.values()) + ["edge_feat_as_str"]
        graph_keys = ["src_id", "dst_id", "feat"]
        df = df.loc[:, dataframe_keys].rename(columns=dict(zip(dataframe_keys,graph_keys)))
        target_nodes = df["src_id"].unique().tolist() + (df["dst_id"]+offset).unique().tolist()

        # Adding new inference edges to the origin graph
        dataset = dgl.data.CSVDataset(self.configs["graph_folder"], force_reload=False)
        g = dataset[0]
        feat = torch.stack([torch.tensor([float(f) for f in f_str.split(",")], dtype=torch.float) for   f_str in df["feat"]])
        g.add_edges(df["src_id"].values, df["dst_id"].values, {'feat': feat}, etype="transaction")
        g.add_edges(df["dst_id"].values, df["src_id"].values, {'feat': feat}, etype="sym_transaction")
        g = dgl.to_homogeneous(g)
        return g, df, target_nodes, offset

    def inference(self, data):
        g, _, target_nodes, _ = data
        sampler = MultiLayerFullNeighborSampler(1)
        #blocks = sampler.sample_blocks()
        dataloader = DataLoader(
            g,
            target_nodes,
            sampler,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=1,
        )
        self.model.eval()
        feat = self.model.emb.weight.data
        with torch.no_grad():
            for l, layer in enumerate(self.model.encoder.layers):
                y = dict()
                with dataloader.enable_cpu_affinity():
                    for input_nodes, output_nodes, blocks in dataloader:
                        h = []
                        for node in input_nodes:
                            if node < feat.shape[0]:
                                h_node = feat[node].unsqueeze(0)
                            else:
                                h_node = torch.rand(1, self.model.hid_size)
                            h += h_node
                        h = torch.stack(h, dim=0)
                        h = layer(blocks[0], h)
                        if l != len(self.model.encoder.layers) - 1:
                            h = F.relu(h)
                        y[output_nodes] = h
                    if feat.shape[0] == self.model.emb.weight.data.shape[0]:
                        padding_matrix = [torch.zeros(1, self.model.hid_size) for k in y if k >= self.model.emb.weight.data.shape[0]]
                        if len(padding_matrix) > 0:
                            padding_matrix = torch.stack(padding_matrix, dim=0)
                            feat = torch.concat([feat, padding_matrix], dim=0)
                    for k, v in y.items():
                        feat[k, :] = v
        return feat
    
    def postprocess(self, node_emb: torch.Tensor, df: pd.DataFrame, offset: int, processed_data: pd.DataFrame):
        node_emb_arr = node_emb.cpu().detach().numpy()
        node_emb_dict = {i:val for i, val in enumerate(node_emb_arr)}
        df.drop(columns=["feat"], inplace=True)
        df["dst_id"] = df["dst_id"] + offset
        df.reset_index(drop=True, inplace=True)
        node_key = ["src_id", "dst_id"]
        for i, node_k in enumerate(node_key):
            emb = pd.DataFrame(df[node_k].map(node_emb_dict).to_list()).add_prefix("n" + str(i) + "_e")
            df = df.join([emb])
            df.drop(columns=[node_k], axis=1, inplace=True)
        processed_data = processed_data.reset_index(drop=True)
        df = pd.concat([processed_data, df], axis=1)
        df = df.drop(columns=['', 'card_id_Idx', 'merchant_id_Idx', 'edge_feat_as_str', 'user', 'card', 'merchant_name', 'is_fraud?', 'split'])
        return df.to_numpy() 

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        output0_dtype = self.output0_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(request, "input__0")

            in_data = in_0.as_numpy().astype(str)

            processed_data = self.preprocess(in_data)
            model_input = self.build_new_graph(processed_data)
            model_output = self.inference(model_input)
            out_data = self.postprocess(model_output, model_input[1], model_input[3], processed_data)

            out_tensor_0 = pb_utils.Tensor("output__0",out_data.astype(output0_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0]
            )
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")