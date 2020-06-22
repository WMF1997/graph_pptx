# train_fc.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dataset import cora_data, num_features, num_classes
from config import device, lr, weight_decay, hidden_features, MAX_EPOCHS

from fc import TwoLayersFC

model = TwoLayersFC()
model, cora_data = model.to(device), cora_data.to(device)

# a simple way of storing result... and save them as .pt ...
# just for the comparision result...
train_acc = []
test_acc = []

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

for epoch in range(MAX_EPOCHS):
    # train
    model.train()
    optimizer.zero_grad()
    loss = F.nll_loss(model(cora_data)[cora_data.train_mask], cora_data.y[cora_data.train_mask])
    loss.backward()
    optimizer.step()
    # test
    model.eval()
    logits, accs = model(cora_data), []
    for _, mask in cora_data('train_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(cora_data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    # printing
    print ('Epoch: {:03d}, loss: {:.4f}, Train: {:.4f}, Test: {:.4f}'.format(epoch, loss, accs[0], accs[1]))
    # appending
    train_acc.append(accs[0])
    test_acc.append(accs[1])

train_acc = torch.tensor(train_acc)
test_acc = torch.tensor(test_acc)


torch.save(train_acc, __file__[:-3]+'_train_result.pt')
torch.save(test_acc, __file__[:-3]+'_test_result.pt')
