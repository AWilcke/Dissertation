import torch.nn as nn
from torch.autograd import Variable
import torch
from torch.nn import functional as F

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
    def __init__(self, bias=False, *args, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(1, 5, 5, bias=bias),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(5, 10, 5, bias=bias),
                nn.MaxPool2d(2),
                nn.ReLU(),
                )
        self.fc = nn.Sequential(
                nn.Linear(160, 16),
                nn.ReLU(),
                nn.Linear(16,1),
                nn.Sigmoid()
                )
    def forward(self, x):
        reshape = x.view(-1, 1, 28, 28)
        y1 = self.conv(reshape)
        flat = y1.view(y1.size(0), -1)
        return self.fc(flat)


class BaseRegressor(nn.Module):
    def __init__(self, bias=[True, True, True, True]):
        '''
        bias :: (list of Bool) : layer-wise presence of bias
        '''

        super().__init__()
        self.bias = bias

    def forward(self, x):
        pass
    
    def tensor_dict(self, weights):
        l = []
        for i, tensor in enumerate(weights):
            d = {}
            if self.bias[i]:
                d['weight'] = tensor[:,:,:-1]
                d['bias'] = tensor[:,:,-1]
            # if no bias, just fill with 0s
            else:
                d['weight'] = tensor
                b = Variable(torch.zeros(tensor.size()[:-1]))
                if tensor.is_cuda:
                    b = b.cuda()
                d['bias'] = b
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
        return l2.div(batch)
    
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
            # print(pred)
            hinge += torch.nn.functional.binary_cross_entropy(pred, y.view(-1,1))

        return hinge.div(regressed_w[0].size(0))

class MLP_Regressor(BaseRegressor):
    def __init__(self, block_size=10, *args, **kwargs):

        def _make_layer(h_dim):
            return nn.Sequential(
                    nn.Linear(h_dim, h_dim),
                    nn.LeakyReLU(0.1),
                    nn.Linear(h_dim, h_dim),
                    )

        super().__init__()
        self.layer0 = _make_layer(785*block_size)
        self.layer1 = _make_layer(101*block_size)
        self.layer2 = _make_layer(101)

        self.block_size = block_size

    def layers(self):
        return [self.layer0,
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

        return [_layer_forward(self.layers()[i], l) for i, l in enumerate(x)]




class ConvRegressor(BaseRegressor):
    
    def __init__(self, filter_size=1, dropout=0, bn=False, *args, **kwargs):

        super().__init__()
        
        assert filter_size % 2 != 0, "Filter size must be odd to ensure size remains same"
        
        self.filter_size = filter_size
        self.padding = filter_size // 2
        self.dropout = dropout
        self.bn = bn

        self.layer0 = self._make_conv()
        self.layer1 = self._make_conv()
        self.layer2 = nn.Sequential(
                nn.Linear(101,101),
                nn.ReLU(),
                nn.Linear(101,101),
                )

    def _make_conv(self):

        if self.bn:
            return nn.Sequential(
                    nn.Conv1d(100, 100, self.filter_size, padding=self.padding),
                    nn.BatchNorm1d(100),
                    nn.Dropout(self.dropout),
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(100, 100, self.filter_size, padding=self.padding),
                    )
        else:
            return nn.Sequential(
                    nn.Conv1d(100, 100, self.filter_size, padding=self.padding),
                    nn.Dropout(self.dropout),
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(100, 100, self.filter_size, padding=self.padding),
                    )


    def layers(self):
        return [self.layer0,
                self.layer1,
                self.layer2,
                ]

    def forward(self, x):
        return [self.layers()[i](l) for i, l in enumerate(x)]

class ConvNetRegressor(BaseRegressor):
    
    def __init__(self, bias=[True]*4, dropout=0, activation='lrelu', bn=True, *args, **kwargs):

        super().__init__(bias=bias)

        self.dropout = dropout
        self.activation = nn.LeakyReLU(0.1) if activation == 'lrelu' else nn.Tanh()
        self.bn = bn

        self.layer_1 = self._make_layer(5*(25+bias[0]))
        self.layer_2 = self._make_layer(10*(125+bias[1]))
        self.layer_3 = self._make_layer(16*(160+bias[2]))
        self.layer_4 = self._make_layer(1*(16+bias[3]))

    def layers(self):
        return [self.layer_1,
                self.layer_2,
                self.layer_3,
                self.layer_4]

    def _make_layer(self, h_dim):
        if self.bn:
            return nn.Sequential(
                    nn.Linear(h_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    self.activation,
                    nn.Dropout(self.dropout),
                    nn.Linear(h_dim, h_dim),
                    )
        else:
            return nn.Sequential(
                    nn.Linear(h_dim, h_dim),
                    self.activation,
                    nn.Dropout(self.dropout),
                    nn.Linear(h_dim, h_dim),
                    )

    def forward(self, x):
        out = []
        for i, weights in enumerate(x):
            reshape = weights.view(weights.size(0), -1)
            regress = self.layers()[i](reshape)
            out.append(regress.view_as(weights))
        return out
    
    def fprop(self, l, ipt, b):
        """
        Forward propagate through the regressed layers, using input ipt and b-th network from l
        """

        ipt = ipt.view(-1, 1, 28, 28)

        # run through convs
        for layer in range(2):
            weights = l[layer]['weight'][b].contiguous()
            weights = weights.view(weights.size(0), -1, 5, 5) # we only have 5 filter size
            ipt = F.conv2d(ipt, weights, l[layer]['bias'][b])
            ipt = F.max_pool2d(ipt, 2)
            ipt = nn.functional.relu(ipt)

        # flatten
        ipt = ipt.view(ipt.size(0), -1)

        # run through fc
        ipt = torch.matmul(ipt, l[2]['weight'][b].transpose(0,1)) + l[2]['bias'][b]
        ipt = nn.functional.relu(ipt)

        # last layer + sigmoid for prob
        pred = torch.matmul(ipt, l[-1]['weight'][b].transpose(0,1)) + l[-1]['bias'][b]
        pred = nn.functional.sigmoid(pred)

        return pred

class LargeConvNetRegressor(ConvNetRegressor):

    def __init__(self, bias=[True]*4, dropout=0, activation='lrelu', bn=True, *args, **kwargs):
        super().__init__(bias=bias, dropout=dropout, activation=activation, bn=bn)

    def _make_layer(self, input_dim):

        h1 = int(1.5*input_dim)
        h2 = int(1.25*input_dim)

        if self.bn:
            return nn.Sequential(
                    nn.Linear(input_dim, h1),
                    nn.BatchNorm1d(h1),
                    self.activation,
                    nn.Dropout(self.dropout),
                    nn.Linear(h1, h2),
                    nn.BatchNorm1d(h2),
                    self.activation,
                    nn.Dropout(self.dropout),
                    nn.Linear(h2, input_dim),
                    nn.BatchNorm1d(input_dim),
                    self.activation,
                    nn.Dropout(self.dropout),
                    nn.Linear(input_dim, input_dim),
                    )
        else:
            return nn.Sequential(
                    nn.Linear(input_dim, h1),
                    self.activation,
                    nn.Dropout(self.dropout),
                    nn.Linear(h1, h2),
                    self.activation,
                    nn.Dropout(self.dropout),
                    nn.Linear(h2, input_dim),
                    self.activation,
                    nn.Dropout(self.dropout),
                    nn.Linear(input_dim, input_dim),
                    )


class ConvConvNetRegressor(ConvNetRegressor):

    def __init__(self, bias=[False, False, True, True], dropout=0, activation='lrelu', bn=True, *args, **kwargs):
        super().__init__(bias=bias, dropout=dropout, activation=activation, bn=bn)

        # overwrite conv layer regressors
        self.layer_1 = self._make_conv_layer(1, 5)
        self.layer_2 = self._make_conv_layer(5, 10)

    def _make_conv_layer(self, input_dim, output_dim, kernel_size=3):

        pad_in = input_dim // kernel_size
        pad_hw = 5 // kernel_size
        kernel_in = kernel_size

        if input_dim == 1:
            pad_in = 0
            kernel_in = 1

        if self.bn:
            out = nn.Sequential(
                    nn.Conv3d(output_dim, output_dim, 
                        (kernel_in, kernel_size, kernel_size),
                        padding=(pad_in, pad_hw, pad_hw)),
                    nn.BatchNorm3d(output_dim),
                    self.activation,
                    nn.Dropout(self.dropout),
                    nn.Conv3d(output_dim, output_dim, 
                        (kernel_in, kernel_size, kernel_size),
                        padding=(pad_in, pad_hw, pad_hw)),
                    nn.BatchNorm3d(output_dim),
                    self.activation,
                    nn.Dropout(self.dropout),
                    nn.Conv3d(output_dim, output_dim, 
                        (kernel_in, kernel_size, kernel_size),
                        padding=(pad_in, pad_hw, pad_hw)),
                    )
        else:
            out = nn.Sequential(
                    nn.Conv3d(output_dim, output_dim, 
                        (kernel_in, kernel_size, kernel_size),
                        padding=(pad_in, pad_hw, pad_hw)),
                    self.activation,
                    nn.Dropout(self.dropout),
                    nn.Conv3d(output_dim, output_dim, 
                        (kernel_in, kernel_size, kernel_size),
                        padding=(pad_in, pad_hw, pad_hw)),
                    self.activation,
                    nn.Dropout(self.dropout),
                    nn.Conv3d(output_dim, output_dim, 
                        (kernel_in, kernel_size, kernel_size),
                        padding=(pad_in, pad_hw, pad_hw)),
                    )
        return out

    def forward(self, x):
        out = []
        for i, weights in enumerate(x):
            if i < 2:
                reshape = weights.view(weights.size(0), weights.size(1), -1, 5, 5)
            else:
                reshape = weights.view(weights.size(0), -1)

            regress = self.layers()[i](reshape)
            out.append(regress.view_as(weights))
        return out


class VAE(nn.Module):
    def __init__(self, input_dim, h1=0.75, h2=0.5, dropout=0):

        super().__init__()

        self.h1 = int(h1 * input_dim)
        self.h2 = int(h2 * input_dim)

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(input_dim, self.h1)
        self.fc21 = nn.Linear(self.h1, self.h2)
        self.fc22 = nn.Linear(self.h1, self.h2)

        self.lrelu = nn.LeakyReLU(0.1)

        self.decoder = nn.Sequential(
                nn.Linear(self.h2, self.h1),
                self.lrelu,
                self.dropout,
                nn.Linear(self.h1, input_dim)
                )

    def encoder(self, x):
        y1 = self.dropout(self.lrelu(self.fc1(x)))
        return self.fc21(y1), self.fc22(y1)

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(.5).exp()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss(self, recon_x, x, mu, logvar):
        BCE = F.mse_loss(recon_x, x.view(x.size(0), -1))

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
 

        return BCE, KLD.mean()


class VAEConvRegressor(ConvNetRegressor):

    def __init__(self, bias=[True]*4, regressor=False, h1=0.75, h2=0.5, dropout=0,*args, **kwargs):
        '''
        regressor :: bool : whether the VAE is being trained as a regressor
                            or as a VAE. In the former case, will only output
                            regressed values, not mu and logvar.
        h1, h2 :: float : scaling factor by which to reduce the input for hidden layers 1 and 2
        '''
        self.h1 = h1
        self.h2 = h2
        super().__init__(bias=bias, dropout=dropout)
        self.regressor = regressor

    def _make_layer(self, input_dim):
        return VAE(input_dim, h1=self.h1, h2=self.h2, dropout=self.dropout)

    def forward(self, x):
        out = []
        mus = []
        logvars = []
        for i, weights in enumerate(x):
            reshape = weights.view(weights.size(0), -1)
            regress, mu, logvar = self.layers()[i](reshape)
            out.append(regress.view_as(weights))
            mus.append(mu)
            logvars.append(logvar)

        if self.regressor:
            return out
        else:
            return out, mus, logvars

    def vae_loss(self, recon_x, x, mus, logvars):
        BCE = Variable(torch.zeros(1))
        KLD = Variable(torch.zeros(1))

        if x[0].is_cuda:
            BCE = BCE.cuda()
            KLD = KLD.cuda()

        # layerwise VAE loss
        for i in range(len(x)):
            i_bce, i_kld = \
                self.layers()[i].loss(recon_x[i], x[i], mus[i], logvars[i])

            BCE += i_bce
            KLD += i_kld

        return BCE, KLD
