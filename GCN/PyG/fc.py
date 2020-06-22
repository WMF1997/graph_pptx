# fc.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import cora_data, num_features, num_classes
from config import device, lr, weight_decay, hidden_features


# we only set 2-fc layers (i.e. the same config, A -> I) to train again. 
class TwoLayersFC(nn.Module):
    def __init__(self):
        super(TwoLayersFC, self).__init__()
        self.fc_1 = nn.Linear(num_features, hidden_features)
        self.fc_2 = nn.Linear(hidden_features, num_classes)
    def forward(self, data):
        x = data.x
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc_1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc_2(x)
        x = F.relu(x)
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    # import torch_geometric
    import torch_geometric.data as gdata
    x = torch.randn(3, num_features)
    edge_index = torch.tensor([
        [0,0,1,1,2],
        [1,2,0,2,1],
    ])
    test_data = gdata.Data(x=x)
    f = TwoLayersFC()
    y = f(test_data)
