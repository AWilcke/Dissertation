import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from dataloader import SVMDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import argparse
from tensorboardX import SummaryWriter
import os

BATCH_SIZE = 4 # do not set to 1, as BatchNorm won't work
NUM_EPOCHS = 10

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
        for i in range(regressed_w.data.shape[0]):
            hinge_loss += torch.mean(torch.clamp(
                1 - torch.matmul(
                    regressed_w[i][:-1], train[i].transpose(0,1)).add(regressed_w[i][-1]),
                min=0))
        
        # average
        hinge_loss.div_(regressed_w.data.shape[0])

        return l2_loss, hinge_loss

def collate_fn(batch):
    # default collate w0, w1
    out = {key: default_collate([d[key] for d in batch]) for key in ('w0','w1')}

    # list of tensors for training samples
    out['train'] = [d['train'] for d in batch]

    return out

def train(args):
    # training datasets
    dataset = SVMDataset(args.w0, args.w1, args.feature, split='train')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
            shuffle=True, num_workers=8, collate_fn=collate_fn)

    # validation datasets
    val_dataset = SVMDataset(args.w0, args.w1, args.feature, split='val')
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
            shuffle=False, num_workers=8, collate_fn=collate_fn)

    writer = SummaryWriter(args.r)
    
    net = SVMRegressor()

    
    start_epoch = 0

    # load model if exists
    if not os.path.exists(args.ckpt):
        os.mkdir(args.ckpt)
    if os.listdir(args.ckpt):

        latest_model = sorted(os.listdir(args.ckpt),
                key=lambda x : int(x.split('.')[0]))[-1]
        print("Restoring from model {}".format(latest_model))
        net.load_state_dict(torch.load(
            os.path.join(args.ckpt, latest_model)))
        
        #update start epoch
        start_epoch = int(latest_model.split('.')[0])
    
    if torch.cuda.is_available():
        net = net.cuda()

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    # lr step after 3 and 6 epochs
    lr_schedule = lr_scheduler.MultiStepLR(optimizer, args.steps, gamma=args.step_gamma)

    for epoch in range(start_epoch, NUM_EPOCHS):
        # for keeping track of loss
        running_l2, running_hinge = 0, 0

        # step lr scheduler so epoch is 1-indexed
        lr_schedule.step() 

        for b, samples in enumerate(dataloader):
            
            global_step = epoch*len(dataloader) + b
            
            if global_step % args.print_every_n == 0:
                print("Global step: {}, Epoch: {}".format(
                    global_step, epoch), end='\r')

            w0 = Variable(samples['w0'].float())
            w1 = Variable(samples['w1'].float())
            train = [Variable(t.float()) for t in samples['train']]

            if torch.cuda.is_available():
                w0 = w0.cuda()
                w1 = w1.cuda()
                train = [t.cuda() for t in train]

            regressed_w = net(w0) # regressed_w[-1] is the intercept

            optimizer.zero_grad()
            
            l2_loss, hinge_loss = net.loss(regressed_w, w1, train)
            running_l2 += l2_loss.data[0]
            running_hinge += hinge_loss.data[0]

            total_loss  = l2_loss + args.lr_weight * hinge_loss

            total_loss.backward()
            optimizer.step()
            
            # write to tensorboard
            if global_step % args.write_every_n == 0 and global_step != 0:

                writer.add_scalar('total_loss/train', 
                        (running_l2 + args.lr_weight * running_hinge)/args.write_every_n,
                        global_step)
                writer.add_scalar('l2_loss/train',
                        running_l2/args.write_every_n, global_step)
                writer.add_scalar('hinge_loss/train',
                        running_hinge/args.write_every_n, global_step)
                writer.add_scalar('learning_rate',
                        optimizer.state_dict()['param_groups'][0]['lr'],
                        global_step)

                running_l2, running_hinge = 0, 0

            # run validation cycle
            if global_step % args.validate_every_n == 0:
                
                # clear up space on gpu
                del w0, w1, train
                
                net.train(False)

                val_l2, val_hinge = 0, 0
                for val_sample in val_dataloader:

                    w0_val = Variable(val_sample['w0'].float())
                    w1_val = Variable(val_sample['w1'].float())
                    train_val = [Variable(t.float()) for t in val_sample['train']]

                    if torch.cuda.is_available():
                        w0_val = w0_val.cuda()
                        w1_val = w1_val.cuda()
                        train_val = [t.cuda() for t in train_val]

                    regressed_val = net(w0_val)

                    val_l2_loss, val_hinge_loss = net.loss(regressed_val, w1_val, train_val)

                    val_l2 += val_l2_loss.data[0]
                    val_hinge += val_hinge_loss.data[0]

                writer.add_scalar('l2_loss/val', val_l2/len(val_dataloader),
                        global_step)
                writer.add_scalar('hinge_loss/val', val_hinge/len(val_dataloader),
                        global_step)
                writer.add_scalar('total_loss/val', 
                        (val_l2 + args.lr_weight * val_hinge)/len(val_dataloader),
                        global_step)

                net.train()
            
        

        # save model
        if (epoch + 1) % args.save_every_n == 0:

            torch.save(net.state_dict(),
                    os.path.join(args.ckpt, "{}.ckpt".format(epoch+1))
                    )

            # remove old models
            models = [os.path.join(args.ckpt, f) for 
                    f in os.listdir(args.ckpt) if ".ckpt" in f]
            models = sorted(models, 
                    key=lambda x : int(os.path.basename(x).split('.')[0]),
                    reverse=True)

            while len(models) > args.n_models_to_keep:
                os.remove(models.pop())



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # io args
    parser.add_argument('-w0')
    parser.add_argument('-w1')
    parser.add_argument('-f', dest='feature')
    parser.add_argument('-r','--runs', dest='r')
    parser.add_argument('--ckpt')

    # training args
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr_weight', type=float, default=1)
    parser.add_argument('--steps', nargs='+', type=int, default=[3,6])
    parser.add_argument('--step_gamma', type=float, default=0.1)

    # logging args
    parser.add_argument('--write_every_n', type=int, default=500)
    parser.add_argument('--print_every_n', type=int, default=10)
    parser.add_argument('--validate_every_n', type=int, default=5000)
    parser.add_argument('--save_every_n', type=int, default=1)
    parser.add_argument('--n_models_to_keep', type=int, default=3)
    args = parser.parse_args()

    train(args)
