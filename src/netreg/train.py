import torch
import torch.optim as optim
from torch.autograd import Variable
from dataloader import MLP_Dataset
from torch.utils.data import DataLoader
from models import MLP_100, MLP_Regressor
from utils import collate_fn
import argparse
from tensorboardX import SummaryWriter
import os
import utils

BATCH_SIZE = 64 # do not set to 1, as BatchNorm won't work
NUM_EPOCHS = 100

model_dict = {
        'mlp_100': MLP_100,
        'mlp_reg': MLP_Regressor,
        }

def train(args):
    # training datasets
    dataset = MLP_Dataset(args.w0, args.w1, train=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
            shuffle=True, num_workers=0, collate_fn=collate_fn)

    # validation datasets
    val_dataset = MLP_Dataset(args.w0, args.w1, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
            shuffle=False, num_workers=0, collate_fn=collate_fn)

    writer = SummaryWriter(args.r)

    # create model
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print("Using {} GPUs".format(n_gpu))
    else:
        n_gpu = 0
    
    
    net = model_dict[args.net](n_gpu=n_gpu)

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

    if args.optimiser == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=tuple(args.betas))
    elif args.optimiser == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr)
    elif args.optimiser == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        raise Exception("Optimiser type not supported : {}".format(args.optimiser))

    log_dic = {
            'hinge':0,
            'l2':0,
            }

    # create variables to store everything
    w0 = [torch.FloatTensor(*((BATCH_SIZE,) + x.size())) for x in dataset[0]['w0']]
    w1 = [torch.FloatTensor(*((BATCH_SIZE,) + x.size())) for x in dataset[0]['w1']]

    # cast to gpu
    if torch.cuda.is_available():
        w0 = [x.cuda() for x in w0]
        w1 = [x.cuda() for x in w1]

    # wrap in variables
    w0 = [Variable(x) for x in w0]
    w1 = [Variable(x) for x in w1]

    for epoch in range(start_epoch, NUM_EPOCHS):

        for b, samples in enumerate(dataloader):
            
            global_step = epoch*len(dataloader) + b

            # copy data to variables
            for a, b in zip(w0, samples['w0']):
                a.copy_(b)
            for a, b in zip(w1, samples['w1']):
                a.copy_(b)
            
            # dynamically create train and labels, as size varies
            train = []
            labels = []

            for t, l in samples['train']:
                train.append(Variable(t.float()))
                labels.append(Variable(l.float()))

            if torch.cuda.is_available():
                train = [t.cuda() for t in train]
                labels = [l.cuda() for l in labels]

            regressed_w = net(w0) # regressed_w[-1] is the intercept

            optimizer.zero_grad()
            
            l2_loss = net.l2_loss(regressed_w, w1)
            hinge_loss = net.perf_loss(regressed_w, train, labels)

            log_dic['l2'] += l2_loss.data[0]
            log_dic['hinge'] += hinge_loss.data[0]

            total_loss  = l2_loss + args.lr_weight * hinge_loss

            total_loss.backward()
            optimizer.step()

            # write to tensorboard
            if global_step % args.write_every_n == 0 and global_step != 0:

                writer.add_scalar('total_loss', 
                        (log_dic['l2'] + args.lr_weight * log_dic['hinge'])/args.write_every_n,
                        global_step)
                writer.add_scalar('l2',
                        log_dic['l2']/args.write_every_n, global_step)
                writer.add_scalar('hinge',
                        log_dic['hinge']/args.write_every_n, global_step)
                writer.add_scalar('learning_rate',
                        optimizer.state_dict()['param_groups'][0]['lr'],
                        global_step)

                log_dic['l2'], log_dic['hinge'] = 0, 0

            ########################
            ###### VALIDATION ######
            ########################

            # get validation metrics for G/C
            
            if global_step % args.validate_every_n == 0:

                utils.validation_metrics(net, val_dataloader, writer, global_step)

            if global_step % args.classif_every_n == 0 and global_step != 0:

                utils.check_performance(net, val_dataset, y, writer, args, global_step)

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
    parser.add_argument('-l', dest='labels')
    parser.add_argument('-r','--runs', dest='r')
    parser.add_argument('--ckpt')

    # architecture args
    parser.add_argument('--gen_name',type=str, default='standard')
    parser.add_argument('--slope', type=float, default=0.01,
            help='negative slope for LeakyRelu')
    parser.add_argument('--dropout', type=float, default=0,
            help='dropout probability for regressor')

    # training args
    parser.add_argument('--optimiser', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--betas', nargs='+', type=float, default=[0.9, 0.999])
    parser.add_argument('--lr_weight', type=float, default=1)
    # parser.add_argument('--steps', nargs='+', type=int, default=[3,6])
    # parser.add_argument('--step_gamma', type=float, default=0.1)
    parser.add_argument('--square_hinge', action='store_true')

    # logging args
    parser.add_argument('--write_every_n', type=int, default=500)
    parser.add_argument('--validate_every_n', type=int, default=5000)
    parser.add_argument('--classif_every_n', type=int, default=10000)
    parser.add_argument('--save_every_n', type=int, default=1)
    parser.add_argument('--n_models_to_keep', type=int, default=3)
    args = parser.parse_args()

    train(args)
