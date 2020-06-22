# gcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
import torch_geometric.nn as gnn

from dataset import num_features, num_classes
from config import hidden_features

class GCNNet(nn.Module):
    def __init__(self):
        super(GCNNet, self).__init__()
        self.gcn_1 = gnn.GCNConv(in_channels=num_features, out_channels=hidden_features)
        self.gcn_2 = gnn.GCNConv(in_channels=hidden_features, out_channels=num_classes)

    def forward(self, data):
        # NOTE(WMF): we can change the dataset much more easily
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, training=self.training)
        x = F.relu(self.gcn_1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.gcn_2(x, edge_index))
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    import torch_geometric.data as gdata
    x = torch.randn(3, num_features)
    edge_index = torch.tensor([
        [0,0,1,1,2,2], 
        [1,2,0,2,0,1]
    ])
    data = gdata.Data(x=x, edge_index=edge_index)
    gcn = GCNNet()
    y = gcn(data)

