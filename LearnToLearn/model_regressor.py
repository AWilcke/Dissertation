import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from dataloader import SVMDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import argparse
from tensorboardX import SummaryWriter

BATCH_SIZE = 1
NUM_EPOCHS = 1000

class SVMRegressor(nn.Module):

    def __init__(self):
        super(SVMRegressor, self).__init__()
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
        for i in range(BATCH_SIZE):
            hinge_loss += torch.mean(torch.clamp(
                1 - torch.matmul(
                    regressed_w[i][:-1], train[i].transpose(0,1)).add(regressed_w[i][-1]),
                min=0))
        
        # average
        hinge_loss.div_(BATCH_SIZE)

        return l2_loss, hinge_loss

def collate_fn(batch):
    # default collate w0, w1
    out = {key: default_collate([d[key] for d in batch]) for key in ('w0','w1')}

    # list of tensors for training samples
    out['train'] = [d['train'] for d in batch]

    return out

def train(args):
    dataset = SVMDataset(args.w0, args.w1, args.feature, split='train')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
            shuffle=True, num_workers=BATCH_SIZE, collate_fn=collate_fn)

    writer = SummaryWriter(args.r)
    
    net = SVMRegressor()
    
    if torch.cuda.is_available():
        net = net.cuda()

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(NUM_EPOCHS):
        # for keeping track of loss
        running_total, running_l2, running_hinge = 0, 0, 0

        for b, samples in enumerate(dataloader):
            
            if b % args.print_every_n == 0:
                print("Global step: {}, Epoch: {}".format(
                    epoch*len(dataloader) + b, epoch), end='\r')

            w0 = samples['w0'].float()
            w1 = samples['w1'].float()
            train = [t.float() for t in samples['train']]

            if torch.cuda.is_available():
                w0 = w0.cuda()
                w1 = w1.cuda()
                train = [t.cuda() for t in train]

            w0 = Variable(w0)
            w1 = Variable(w1)
            train = [Variable(t) for t in train]

            regressed_w = net(w0) # regressed_w[-1] is the intercept

            optimizer.zero_grad()
            
            l2_loss, hinge_loss = net.loss(regressed_w, w1, train)
            running_l2 += l2_loss.data[0]
            running_hinge += hinge_loss.data[0]

            total_loss  = args.lr * (l2_loss + args.lr_weight * hinge_loss)
            running_total += total_loss.data[0]

            total_loss.backward()
            optimizer.step()
            
            # write to tensorboard
            if b % args.write_every_n == 0 and b != 0:
                global_step = epoch*len(dataloader) + b
                writer.add_scalar('train_total_loss', 
                        running_total/args.write_every_n, global_step)
                writer.add_scalar('train_l2_loss',
                        running_l2/args.write_every_n, global_step)
                writer.add_scalar('train_hinge_loss',
                        running_hinge/args.write_every_n, global_step)

                running_total, running_l2, running_hinge = 0, 0, 0

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-w0')
    parser.add_argument('-w1')
    parser.add_argument('-f', dest='feature')
    parser.add_argument('-r','--runs', dest='r')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_weight', type=float, default=1)
    parser.add_argument('--write_every_n', type=int, default=500)
    parser.add_argument('--print_every_n', type=int, default=10)
    args = parser.parse_args()

    train(args)
