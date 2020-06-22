# config.py

import torch

# hardware
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training details
batch_size = 1

hidden_features = 16

MAX_EPOCHS = 200
lr = 1e-2
weight_decay = 5e-3

