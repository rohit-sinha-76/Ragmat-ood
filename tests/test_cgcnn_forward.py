"""Test CGCNN forward."""
import pytest
import torch
from torch_geometric.data import Data, Batch
from ragmat.encoders.cgcnn import CGCNNEncoder

def test_cgcnn_forward():
    encoder = CGCNNEncoder(node_dim=92, edge_dim=40, hidden_dim=64, n_conv_layers=2)
    # Dummy data
    x = torch.zeros(10, 92)
    x[:, 0] = 1.0 # 10 H atoms
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.randn(4, 40)
    y = torch.tensor([[0.5]])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    batch = Batch.from_data_list([data, data])
    
    pred, emb = encoder(batch)
    assert pred.shape == (2, 1)
    assert emb.shape == (2, 64)
    assert not torch.isnan(pred).any()
    assert not torch.isnan(emb).any()
