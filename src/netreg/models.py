import torch.nn as nn
from torch.autograd import Variable
import torch

class MLP_100(nn.Module):

    def __init__(self, *args, **kwargs):
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

class ConvNet(nn.Module):
    def __init__(self, *args, **kwargs):
        self.conv = nn.Sequential(
                nn.Conv2d(3, 5, 5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(5, 10, 5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                )
        self.fc = nn.Sequential(
                nn.Linear(160, 16),
                nn.ReLU(),
                nn.Linear(16,1),
                )
    def forward(self, x):
        y1 = self.conv(x)
        flat = y1.view(y1.size(0), -1)
        return self.fc(flat)


class BaseRegressor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
    
    def tensor_dict(self, weights):
        l = []
        for i, tensor in enumerate(weights):
            d = {}
            d['weight'] = tensor[:,:,:-1]
            d['bias'] = tensor[:,:,-1]
            l.append(d)

        return l

    def fprop(self, l, ipt, b):
        """
        Forward propagate through the regressed layers, using input ipt and b-th network from l
        """
        # run through all but last layer
        for layer in range(len(l)-1):
            ipt = torch.matmul(ipt, l[layer]['weight'][b].transpose(0,1)) + l[layer]['bias'][b]
            ipt = nn.functional.relu(ipt)
        
        # last layer + sigmoid for prob
        pred = torch.matmul(ipt, l[-1]['weight'][b].transpose(0,1)) + l[-1]['bias'][b]
        pred = nn.functional.sigmoid(pred)

        return pred

    def l2_loss(self, regressed_w, w1):

        l2 = Variable(torch.zeros(1))
        l2 = l2.cuda() if w1[0].is_cuda else l2
        batch = w1[0].size()[0]

        for regress, target in zip(regressed_w, w1):
            l2 += .5 * torch.mean(
                    torch.norm(
                        regress.view(batch,-1) - target.view(batch, -1),
                        p=2, dim=1)).pow(2)
        return l2
    
    def perf_loss(self, regressed_w, train, labels):

        hinge = Variable(torch.zeros(1))
        hinge = hinge.cuda() if regressed_w[0].is_cuda else hinge
        
        # easy access to stuff
        l = self.tensor_dict(regressed_w)

        # iterate through batch
        for b in range(regressed_w[0].size()[0]):
            ipt = train[b]
            y = labels[b]
            pred = self.fprop(l, ipt, b)
            hinge += torch.nn.functional.binary_cross_entropy(pred, y.view(-1,1))

        return hinge

class MLP_Regressor(BaseRegressor):
    def __init__(self, block_size=10, *args, **kwargs):

        def _make_layer(h_dim):
            return nn.Sequential(
                    nn.Linear(h_dim, h_dim),
                    nn.ReLU(),
                    nn.Linear(h_dim, h_dim),
                    )

        super().__init__()
        self.layer0 = _make_layer(785*block_size)
        self.layer1 = _make_layer(101*block_size)
        self.layer2 = _make_layer(101)

        self.block_size = block_size

        self.layers = [self.layer0,
                self.layer1,
                self.layer2,
                ]

    def forward(self, x):
        def _layer_forward(layer, x):
            batch, hidden_dim, input_dim = x.size()

            block_size = 1 if hidden_dim == 1 else self.block_size
            
            assert hidden_dim % block_size == 0, "hidden_dim({}) is not a multiple of block_size ({})".format(hidden_dim, block_size)

            num_blocks = hidden_dim // block_size # number of blocks to concat to get right out size
            reshaped = x.view(batch, num_blocks, block_size*input_dim)
            reshaped = reshaped.transpose(0,1) # get batch in there

            out = []

            for i in range(num_blocks):
                regressed = layer(reshaped[i,:,:].contiguous().view(batch, -1))
                regressed = regressed.view(batch, block_size, input_dim)
                out.append(regressed)

            return torch.cat(out, dim=1)

        return [_layer_forward(self.layers[i], l) for i, l in enumerate(x)]




class ConvRegressor(BaseRegressor):
    
    def __init__(self, filter_size=1, *args, **kwargs):

        super().__init__()
        
        assert filter_size % 2 != 0, "Filter size must be odd to ensure size remains same"

        padding = filter_size // 2

        self.layer_1 = nn.Conv1d(100, 100, filter_size, padding=padding)
        self.layer_2 = nn.Conv1d(100, 100, filter_size, padding=padding)
        self.layer_3 = nn.Sequential(
                nn.Linear(101,101),
                nn.ReLU(),
                nn.Linear(101,101),
                )

        self.layers = [self.layer_1, self.layer_2, self.layer_3]

    def forward(self, x):
        return [self.layers[i](l) for i, l in enumerate(x)]
