import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn import SAGEConv
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler
)


class GraphSAGE(nn.Module):
    def __init__(
        self, in_feats, hidden_size, out_feats, n_layers, activation, aggregator_type
    ):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, hidden_size, aggregator_type))
        # hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConv(hidden_size, hidden_size, aggregator_type))
        # output layer
        self.layers.append(
            SAGEConv(hidden_size, out_feats, aggregator_type)
        )  # activation None

    def forward(self, graphs, inputs):
        h = inputs
        for l, (layer, block) in enumerate(zip(self.layers, graphs)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
        return h

class Model(nn.Module):
    def __init__(self, vocab_size, hid_size, n_layers):
        super().__init__()

        self.hid_size = hid_size

        # node embedding
        self.emb = torch.nn.Embedding(vocab_size, hid_size)
        # encoder is a 1-layer GraphSAGE model
        self.encoder = GraphSAGE(hid_size, hid_size, hid_size, n_layers, F.relu, "mean")
        # decoder is a 3-layer MLP
        self.decoder = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )
        # cosine similarity with linear
        # self.predictor=dglnn.EdgePredictor('cos',hid_size,1)

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = self.emb(x)
        h = self.encoder(blocks, h)

        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        # h_pos = self.predictor(h[pos_src], h[pos_dst])
        # h_neg = self.predictor(h[neg_src], h[neg_dst])
        h_pos = self.decoder(h[pos_src] * h[pos_dst])
        h_neg = self.decoder(h[neg_src] * h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, device, batch_size):
        """Layer-wise inference algorithm to compute GNN node embeddings."""
        # feat = g.ndata['feat']
        # use pretrained embedding as node features
        feat = self.emb.weight.data
        sampler = MultiLayerFullNeighborSampler(1)
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device
        # compure representations layer by layer
        for l, layer in enumerate(self.encoder.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            # within a layer iterate over nodes in batches
            with dataloader.enable_cpu_affinity():
                for input_nodes, output_nodes, blocks in tqdm.tqdm(
                    dataloader, desc="Inference"
                ):
                    x = feat[input_nodes]
                    h = layer(blocks[0], x)
                    if l != len(self.encoder.layers) - 1:
                        h = F.relu(h)
                    y[output_nodes] = h.to(buffer_device)
                feat = y
        return y
    
class GNNServeModel(Model):
    def inference(self, g: dgl.DGLGraph, target_nodes: list, batch_size: int=1):
        """
        Inference on the entire graph.
        """
        feat = self.model.emb.weight.data
        sampler = MultiLayerFullNeighborSampler(1)
        #blocks = sampler.sample_blocks()
        dataloader = DataLoader(
            g,
            target_nodes,
            sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
        )
        buffer_device = torch.device("cpu")

        feat = self.model.emb.weight.data
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
                    y[output_nodes] = h.to(buffer_device)
                if feat.shape[0] == self.model.emb.weight.data.shape[0]:
                    padding_matrix = [torch.zeros(1, self.model.hid_size) for k in y if k >= self.model.emb.weight.data.shape[0]]
                    if len(padding_matrix) > 0:
                        padding_matrix = torch.stack(padding_matrix, dim=0)
                        feat = torch.concat([feat, padding_matrix], dim=0)
                for k, v in y.items():
                    feat[k, :] = v
        return feat