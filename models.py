import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ERMModel(nn.Module):
    def __init__(self):
        super(ERMModel, self).__init__()
        # self.linear = nn.Linear(10, 100) # Input is 10d and output is 1d
        # self.relu = nn.ReLU()
        self.output = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x = self.relu(self.linear(x))
        x = self.sigmoid(self.output(x))
        return x

