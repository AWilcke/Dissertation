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

class MLP_Regressor(nn.Module):
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
            hidden_dim, input_dim = x.size()
            
            if hidden_dim % self.block_size != 0:
                raise Exception("Incompatible block size")

            num_blocks = hidden_dim // self.block_size # number of blocks to concat to get right out size
            reshaped = x.view(num_blocks, self.block_size, input_dim)

            out = []

            for i in range(num_blocks):
                out.append(layer(reshaped[i,:,:].view(-1)).view(self.block_size, input_dim))

            return torch.cat(out, dim=1)

        return [_layer_forward(self.layers[i], l) for i, l in enumerate(x)]

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
        batch = w1[0].size()[0]

        for regress, target in zip(regressed_w, w1):
            l2 += .5 * torch.mean(
                    torch.norm(
                        regress.view(batch,-1) - target.view(batch, -1),
                        p=2, dim=1)).pow(2)
        return l2
    
    def perf_loss(self, regressed_w, train, labels):

        hinge = Variable(torch.zeros(1))
        
        # easy access to stuff
        l = self.tensor_dict(regressed_w)

        # iterate through batch
        for b in range(regressed_w[0].size()[0]):
            ipt = train[b]
            y = labels[b]
            pred = self.frop(l, ipt, b)
            hinge += torch.nn.functional.binary_cross_entropy(pred, y) 

        return hinge
