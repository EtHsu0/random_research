import torch.nn as nn
import torch.nn.functional as F
import torch
class linearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linearModel, self).__init__()
        self.output = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.output(x))


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, input_dim * 10),
            nn.ReLU(),
            nn.Linear(input_dim * 10, input_dim * 10),
            nn.ReLU(),
            nn.Linear(input_dim * 10, output_dim),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return torch.sigmoid(logits)
