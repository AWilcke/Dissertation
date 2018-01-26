import torch
import torch.nn as nn

class MLP_100(nn.Module):

    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
                nn.Linear(784,100),
                nn.ReLU(),
                nn.Linear(100,100),
                nn.ReLU(),
                nn.Linear(100,1),
                nn.Sigmoid()
                )

    def forward(self, x):
        y = self.main(x)
        return y
