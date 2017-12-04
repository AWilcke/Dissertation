import torch
import torch.nn as nn
from torch.autograd import Variable
import copy

class LayerNorm(nn.Module):

    def __init__(self, weights, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(weights))
        self.beta = nn.Parameter(torch.zeros(weights))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class SVMRegressor(nn.Module):

    def __init__(self, dropout=False, slope=0.01, square_hinge=False, n_gpu=1, tanh=False):
        super(SVMRegressor, self).__init__()
        self.square_hinge = square_hinge
        self.ngpu = n_gpu

        # optional dropout param
        d = [nn.Dropout()] if dropout else []
        t = [nn.Tanh()] if tanh else []
        self.main = nn.Sequential(
            nn.Linear(4097, 6144),
            nn.BatchNorm1d(6144),
            *copy.deepcopy(d),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Linear(6144, 5120),
            nn.BatchNorm1d(5120),
            *copy.deepcopy(d),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Linear(5120, 4097),
            nn.BatchNorm1d(4097),
            *copy.deepcopy(d),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Linear(4097, 4097),
            *copy.deepcopy(t)
            )

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            output = self.main(x)
        return output
    
    def loss(self, regressed_w, w1, train):
        l2_loss = .5 * torch.mean(torch.norm(w1.sub(regressed_w), 2, 1).pow(2))

        hinge_loss = Variable(torch.zeros(1))
        if torch.cuda.is_available():
            hinge_loss = hinge_loss.cuda()

        # calculate summed hinge loss of each regressed svm
        for i in range(regressed_w.data.shape[0]):
            hinge_vector = torch.clamp(
                    1 - torch.matmul(
                        regressed_w[i][:-1], train[i].transpose(0,1))
                    .add(regressed_w[i][-1]),
                    min=0)

            # square the hinge loss if requested
            hinge_vector = torch.clamp(hinge_vector, max=100).pow(2) if self.square_hinge else hinge_vector

            hinge_loss += torch.mean(hinge_vector)
        
        # average
        hinge_loss.div_(regressed_w.data.shape[0])

        return l2_loss, hinge_loss

class Generator8(SVMRegressor):
    def __init__(self, *args, dropout=False, slope=0.01, n_gpu=1, tanh=False, **kwargs):
        super().__init__(dropout=dropout, slope=slope, square_hinge=False, n_gpu=n_gpu)

        # optional dropout param
        d = [nn.Dropout()] if dropout else []
        t = [nn.Tanh()] if tanh else []
        self.main = nn.Sequential(
            nn.Linear(4097, 4097),
            nn.BatchNorm1d(4097),
            *copy.deepcopy(d),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Linear(4097, 6144),
            nn.BatchNorm1d(6144),
            *copy.deepcopy(d),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Linear(6144, 6144),
            nn.BatchNorm1d(6144),
            *copy.deepcopy(d),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Linear(6144, 5120),
            nn.BatchNorm1d(5120),
            *copy.deepcopy(d),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Linear(5120, 5120),
            nn.BatchNorm1d(5120),
            *copy.deepcopy(d),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Linear(5120, 4097),
            nn.BatchNorm1d(4097),
            *copy.deepcopy(d),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Linear(4097, 4097),
            nn.BatchNorm1d(4097),
            *copy.deepcopy(d),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Linear(4097, 4097),
            *copy.deepcopy(t)
            )

class UGen(nn.Module):

    def __init__(self, *args, tanh=False, **kwargs):
        super().__init__()
        self.fc1 = self.make_fc(4097, 4096)
        self.fc2 = self.make_fc(4096, 2048)
        self.fc3 = self.make_fc(2048, 1024)
        self.fc4 = self.make_fc(1024, 2048)
        self.fc5 = self.make_fc(4096, 4096)
        self.fc6 = self.make_fc(8192, 4097)
        self.fc7 = self.make_fc(8194, 4097)
        self.fc8 = self.make_fc(4097, 4097)

        self.tanh = tanh

    def make_fc(self, in_dim, out_dim):
        return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.Dropout(),
                nn.LeakyReLU(negative_slope=0.01, inplace=True)
                )

    def forward(self, x):

        # normal layers
        out_1 = self.fc1(x)
        out_2 = self.fc2(out_1)
        out_3 = self.fc3(out_2)
        out_4 = self.fc4(out_3)
        # skip connections
        out_5 = self.fc5(torch.cat([out_4, out_2], dim=1))
        out_6 = self.fc6(torch.cat([out_5, out_1], dim=1))
        out_7 = self.fc7(torch.cat([out_6, x], dim=1))
        out_8 = self.fc8(out_7)
        if self.tanh:
            out_8 = nn.Tanh(out_8)
        return out_8

    def loss(self, regressed_w, w1, train):
        l2_loss = .5 * torch.mean(torch.norm(w1.sub(regressed_w), 2, 1).pow(2))

        hinge_loss = Variable(torch.zeros(1))
        if torch.cuda.is_available():
            hinge_loss = hinge_loss.cuda()

        # calculate summed hinge loss of each regressed svm
        for i in range(regressed_w.data.shape[0]):
            hinge_vector = torch.clamp(
                    1 - torch.matmul(
                        regressed_w[i][:-1], train[i].transpose(0,1))
                    .add(regressed_w[i][-1]),
                    min=0)

            hinge_loss += torch.mean(hinge_vector)
        
        # average
        hinge_loss.div_(regressed_w.data.shape[0])

        return l2_loss, hinge_loss


class Critic3(nn.Module):

    def __init__(self, n_gpu=1, gp=True):
        super().__init__()
        
        norm_layer = LayerNorm if gp else nn.BatchNorm1d

        self.main = nn.Sequential(
                nn.Linear(4097, 4097),
                norm_layer(4097),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Linear(4097, 512),
                norm_layer(512),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Linear(512, 512),
                norm_layer(512),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Linear(512,1)
                )
        self.ngpu = n_gpu

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            output = self.main(x)
        return output

class Critic4(nn.Module):

    def __init__(self, n_gpu=1, gp=True):
        super().__init__()
        
        norm_layer = LayerNorm if gp else nn.BatchNorm1d

        self.main = nn.Sequential(
                nn.Linear(4097, 4097),
                norm_layer(4097),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(),
                nn.Linear(4097, 4097),
                norm_layer(4097),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(),
                nn.Linear(4097, 512),
                norm_layer(512),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(),
                nn.Linear(512, 512),
                norm_layer(512),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(),
                nn.Linear(512,1)
                )
        self.ngpu = n_gpu

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            output = self.main(x)
        return output

class Critic8(Critic4):
    def __init__(self, n_gpu=1, gp=True):
        super().__init__(n_gpu=n_gpu, gp=gp)
        norm_layer = LayerNorm if gp else nn.BatchNorm1d
        self.main = nn.Sequential(
                nn.Linear(4097, 4097),
                norm_layer(4097),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(),
                nn.Linear(4097, 4097),
                norm_layer(4097),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(),
                nn.Linear(4097, 4097),
                norm_layer(4097),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(),
                nn.Linear(4097, 4097),
                norm_layer(4097),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(),
                nn.Linear(4097, 512),
                norm_layer(512),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(),
                nn.Linear(512, 512),
                norm_layer(512),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(),
                nn.Linear(512, 512),
                norm_layer(512),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(),
                nn.Linear(512, 512),
                norm_layer(512),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(),
                nn.Linear(512,1)
                )
