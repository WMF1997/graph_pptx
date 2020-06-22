# ImageConv.py
import torch
import torch.nn.functional as F
# convolution operation
w = torch.randn(6, 1, 5, 5)         # out_channel, in_channel, h, w
x = torch.randn(2, 1, 28, 28)       # N, in_channel, H, W
y = F.conv2d(x, w, stride=[1,1], padding=[0,0])    # N, out_channel, H-h+1, W-w+1

