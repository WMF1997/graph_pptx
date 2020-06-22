# plot_result.py
import torch
import numpy as np
import matplotlib.pyplot as plt

fc_test = torch.load('train_fc_test_result.pt').numpy()
gcn_test = torch.load('train_gcn_test_result.pt').numpy()

plt.figure()
l1, = plt.plot(gcn_test, color='#EE0000')
l2, = plt.plot(fc_test, color='#006666')
plt.legend([l1, l2], ['gcn', 'fc'])
plt.title('Cora Dataset test result')
plt.grid('on')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()
