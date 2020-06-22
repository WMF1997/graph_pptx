# dataset.py

import torch

import torch_geometric
import torch_geometric.data as gdata
import torch_geometric.datasets as gdatasets
import torch_geometric.transforms as gtransforms

from config import batch_size

transform = gtransforms.AddSelfLoops()

# test if transform works
# cora = gdatasets.KarateClub(transform=transform)
# cora_loader = gdata.DataLoader(cora, batch_size=1, shuffle=True)

cora = gdatasets.Planetoid(root='./Planetoid/Cora', name='Cora', transform=transform)
cora_data = cora[0]

cora_data.train_mask = torch.zeros(cora_data.num_nodes, dtype=torch.uint8)
cora_data.train_mask[:cora_data.num_nodes-1000] = 1
cora_data.val_mask = None
cora_data.test_mask = torch.zeros(cora_data.num_nodes, dtype=torch.uint8)
cora_data.test_mask[cora_data.num_nodes-500:] = 1

# We only need the train part of the graph to train.

num_features = cora.num_features
num_classes = cora.num_classes

# information about the given dataset/batch
# if __name__ == '__main__':
#     for graph_batch in cora_loader:
#         pass