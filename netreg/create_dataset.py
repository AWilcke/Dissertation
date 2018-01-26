import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models import MLP_100
from argparse import ArgumentParser
from dataset import MNISTbyClass

BATCH_SIZE = 2
VAL_BATCH = 250
NUM_EPOCHS = 100

def main(args):
    data = MNISTbyClass('data/mnist','data/mnist/index.pickle', args.label, args.n, args.train, True)

    val = MNISTbyClass('data/mnist','data/mnist/index.pickle', args.label, 1000, args.train, False)

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
    y = Variable(torch.FloatTensor(BATCH_SIZE))

    x_val = Variable(torch.FloatTensor(VAL_BATCH, 28*28))
    y_val = Variable(torch.FloatTensor(VAL_BATCH))

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
            y.data.copy_(labels)

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
                y_val.data.copy_(labels)

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

            val_text = "Val: {:.2e} Acc: {:.3f}".format(
                    log['V_err']/log['V_count'], acc)

            print("Epoch {:2d} | {} | {} |".format(epoch, train_text, val_text))

        # check if should early stop
        max_epoch, _ = max_val

        if epoch - max_epoch > args.patience * args.val_interval:
            print("No progress in {} epochs, stopping training."
                    .format(args.patience * args.val_interval))
            break

if __name__ == '__main__':

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
   parser.add_argument('--train', type=bool, default=True)
   parser.add_argument('--label', type=int)

   args = parser.parse_args()

   main(args)
