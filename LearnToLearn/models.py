import torch
import torch.nn as nn
from torch.autograd import Variable

class SVMRegressor(nn.Module):

    def __init__(self, square_hinge=False):
        super(SVMRegressor, self).__init__()
        self.square_hinge = square_hinge
        self.regression = nn.Sequential(
            nn.Linear(4097, 6144),
            nn.BatchNorm1d(6144),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(6144, 5120),
            nn.BatchNorm1d(5120),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(5120, 4097),
            nn.BatchNorm1d(4097),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(4097, 4097)
            )

    def forward(self, x):
        return self.regression(x)
    
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

class Critic(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(4097, 4097),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Linear(4097, 4097),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Linear(4097,1)
                )

    def forward(self, x):
        return self.net(x)
