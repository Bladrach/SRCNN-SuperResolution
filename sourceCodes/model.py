import torch
import torch.nn as nn
import torch.nn.functional as F

"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 9, padding = (4,4))
        self.conv2 = nn.Conv2d(128, 64, 3, padding = (1,1))
        self.conv3 = nn.Conv2d(64, 1, 5, padding = (2,2))


    def forward(self, x):

        x = F.relu(self.conv1(x.float()))
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = self.conv3(x)
        #print(x.shape)
        return x
"""

"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 9, padding = (4,4))
        self.conv2 = nn.Conv2d(64, 32, 1, padding = 0)
        self.conv3 = nn.Conv2d(32, 1, 5, padding = (2,2))


    def forward(self, x):

        x = F.relu(self.conv1(x.float()))
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = self.conv3(x)
        #print(x.shape)
        return x
"""
# Used models
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 9, padding = 4)
        self.conv2 = nn.Conv2d(64, 32, 5, padding = 2)
        self.conv3 = nn.Conv2d(32, 1, 5, padding = 2)


    def forward(self, x):

        x = F.relu(self.conv1(x.float()))
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = self.conv3(x)
        #print(x.shape)
        return x
