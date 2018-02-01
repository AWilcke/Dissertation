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

class MLP_Regressor(nn.Module):
    def __init__(self):

        def _make_layer(h_dim):
            return nn.Sequential(
                    nn.Linear(h_dim, h_dim),
                    nn.ReLU,
                    nn.Linear(h_dim, h_dim),
                    )

        super().__init__()
        self.layer0 = _make_layer(784*100)
        self.layer1 = _make_layer(100*100)
        self.layer2 = _make_layer(100)

        self.layer_dict = {
                0: self.layer0,
                1: self.layer1,
                2: self.layer2,
                }

    def forward(self, x, layer):
        return self.layer_dict[layer](x)
