import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models import MLP_100
from argparse import ArgumentParser
from dataset import MNISTbyClass
import itertools
import os
import pickle
import sys
from pathlib import Path

BATCH_SIZE = 2
VAL_BATCH = 200
NUM_EPOCHS = 50

parser = ArgumentParser()

parser.add_argument('--val_interval', type=int, default=1,
       help='Epoch interval for validation cycle')
parser.add_argument('--patience', type=int, default=10,
       help='Patience for early stopping')

parser.add_argument('--lr', type=float, default=0.001,
       help='Learning rate')
parser.add_argument('--optim',
       choices=['sgd','adam','rms'],
       default='sgd')
parser.add_argument('--momentum', type=float, default=0.9,
       help='Momentum for SGD')
parser.add_argument('--betas', nargs='+', type=float,
       default=[0.9, 0.999])
parser.add_argument('--weight_decay', type=float,
       default=0.)

parser.add_argument('-n', type=int)
parser.add_argument('--val', action='store_false')
parser.add_argument('--label', type=int)

parser.add_argument('--output', type=str, default='/dev/null',
       help='Where to save the model to')
parser.add_argument('--logging', action='store_true')

parser.add_argument('--w1', action='store_true')

def main(args, logging=True):

    if args.w1:
        global BATCH_SIZE
        BATCH_SIZE = 256
        args.n = 5000

    data = MNISTbyClass('data/mnist','data/mnist/index.pickle', args.label, args.n, args.val, True)

    val = MNISTbyClass('data/mnist','data/mnist/index.pickle', args.label, 800, args.val, False)

    loader = DataLoader(data, batch_size=BATCH_SIZE,
            shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val, batch_size=VAL_BATCH,
            shuffle=True, num_workers=0, drop_last=True)

    net = MLP_100()

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, 
                momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, 
                betas=tuple(args.betas), weight_decay=args.weight_decay)
    elif args.optim == 'rms':
        optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr,
                weight_decay=args.weight_decay)

    loss = torch.nn.BCELoss()

    x = Variable(torch.FloatTensor(BATCH_SIZE, 28*28))
    y = Variable(torch.FloatTensor(BATCH_SIZE,1))

    x_val = Variable(torch.FloatTensor(VAL_BATCH, 28*28))
    y_val = Variable(torch.FloatTensor(VAL_BATCH, 1))

    if torch.cuda.is_available():
        net = net.cuda()
        x = x.cuda()
        y = y.cuda()
        x_val = x_val.cuda()
        y_val = y_val.cuda()
    
    # for early stopping
    max_val = (0,0)

    for epoch in range(NUM_EPOCHS):

        # for tracking statistics
        log = {
                'T_err':0,
                'T_count':0,
                'V_err':0,
                'V_count':0,
                'correct':0,
                }

        for images, labels in loader:

            x.data.copy_(images)
            y.data.copy_(labels.view(-1,1))

            optimizer.zero_grad()

            out = net(x)

            error = loss(out, y)
            error.backward()

            optimizer.step()

            log['T_err'] += error.data[0]
            log['T_count'] += y.size(0)

        train_text = "Train: {:.2e}".format(log['T_err']/log['T_count'])
        val_text = ""
        
        if (epoch + 1) % args.val_interval == 0:

            for images, labels in val_loader:
                x_val.data.copy_(images)
                y_val.data.copy_(labels.view(-1,1))

                out = net(x_val)
                pred = (out.data > 0.5).float()
                error = loss(out, y_val)

                log['V_count'] += y_val.size(0)
                log['correct'] += (pred == y_val.data).sum()
                log['V_err'] += error.data[0]

            acc = log['correct']/log['V_count']
                

            # update early stopping criterion
            max_epoch, max_acc = max_val
            if acc > max_acc:
                max_val = (epoch, acc)

            val_text = "Val: {:.2e} | Acc: {:.3f}".format(
                    log['V_err']/log['V_count'], acc)

            if args.logging:
                print("Epoch {:2d} | {} | {} |".format(epoch, train_text, val_text))

        # check if should early stop
        max_epoch, _ = max_val

        if epoch - max_epoch > args.patience * args.val_interval:
            if args.logging:
                print("No progress in {} epochs, stopping training."
                        .format(args.patience * args.val_interval))
            break

    out = {
            'weights': {key: weights.cpu() for key, weights in net.state_dict()},
            'correct_i': data.correct_i,
            'wrong_i': data.wrong_i,
            }

    with open(args.output,'wb') as f:
        pickle.dump(out,f)

if __name__ == '__main__':

    # optims = [
            # 'sgd --momentum 0',
            # 'sgd --momentum 0.5',
            # 'sgd --momentum 0.9',
            # 'adam',
            # 'rms'
            # ]
    # ns = range(1, 51)
    # lrs = [1e-2, 1e-3]
    # wds = [0, 1e-3]

    optims = [
            'rms'
            ]
    ns = [5000]
    lrs = [1e-3]
    wds = [1e-3]
    labels = range(10)
    # labels = range(2,10)
    # labels = [0,1]

    split = 'train' if parser.parse_args().val else 'val'
    stage = 'w1' if parser.parse_args().w1 else 'w0'

    for label in labels:
        count = 0

        # split = 'val' if label in {0,1} else 'train' #hacky overwriting

        for optim, n, lr, wd in itertools.product(optims, ns, lrs, wds):
            # format
            name = '{}_{}'.format(label, count)
            parent_d = Path('data/mnist') / stage / split / str(label)
            parent_d.mkdir(parents=True, exist_ok=True)

            output = parent_d / name

            if not os.path.exists(output):
                arg_str = '--optim {} -n {} --lr {} --weight_decay {} --label {} --output {} '\
                        .format(optim, n, lr, wd, label, output)

                # pass in cli args too
                if len(sys.argv) > 1:
                    arg_str += ' '.join(sys.argv[1:])

                print(arg_str)

                # train
                args = parser.parse_args(arg_str.split())
                main(args)

            count += 1

    # args = parser.parse_args()

    # main(args, False)
