import torch
import torch.optim as optim
from torch.autograd import Variable
from dataset import MLP_Dataset
from torch.utils.data import DataLoader
from models import ConvNet, VAEConvRegressor
import argparse
from tensorboardX import SummaryWriter
import os
import scoring
from tqdm import tqdm
from utils import collate_fn

BATCH_SIZE = 64  # do not set to 1, as BatchNorm won't work
NUM_EPOCHS = 500

model_dict = {
        'conv_net': ConvNet,
        'vae': VAEConvRegressor,
        }


def train(args):
    # training datasets
    dataset = MLP_Dataset(args.w0, args.w1,
            args.mnist, train=True, extended=args.extended)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
            shuffle=True, num_workers=0, collate_fn=collate_fn)

    print(f"Train: {len(dataset)}")

    # validation datasets
    val_dataset = MLP_Dataset(args.w0, args.w1,
            args.mnist, train=False, extended=args.extended)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
            shuffle=False, num_workers=0, collate_fn=collate_fn)
    print(f"Val: {len(val_dataset)}")

    writer = SummaryWriter(args.r)

    # create model
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print("Using {} GPUs".format(n_gpu))
    else:
        print("Using CPU")
        n_gpu = 0

    net = model_dict[args.net]()
    print(net)

    start_epoch = 0

    # load model if exists
    if not os.path.exists(args.ckpt):
        os.mkdir(args.ckpt)

    if os.listdir(args.ckpt):
        latest_model = sorted(os.listdir(args.ckpt),
                key=lambda x: int(x.split('.')[0]))[-1]
        print("Restoring from model {}".format(latest_model))
        net.load_state_dict(torch.load(
            os.path.join(args.ckpt, latest_model)))

        # update start epoch
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
            'BCE': 0,
            'KLD': 0,
            }

    # create variables to store everything
    w0 = [torch.FloatTensor(*((BATCH_SIZE,) + x.size())) for x in dataset[0]['w0']]
    w1 = [torch.FloatTensor(*((BATCH_SIZE,) + x.size())) for x in dataset[0]['w1']]


    # wrap in variables
    w0 = [Variable(x) for x in w0]
    w1 = [Variable(x) for x in w1]

    # cast to gpu
    if torch.cuda.is_available():
        w0 = [x.cuda() for x in w0]
        w1 = [x.cuda() for x in w1]

    for epoch in tqdm(range(start_epoch, NUM_EPOCHS)):

        for b, samples in enumerate(tqdm(dataloader)):

            global_step = epoch*len(dataloader) + b

            # copy data to variables
            for a, b in zip(w0, samples['w0']):
                a.data.copy_(b)
            for a, b in zip(w1, samples['w1']):
                a.data.copy_(b)

            # dynamically create train and labels, as size varies
            train = []
            labels = []

            for t, l in samples['train']:
                train.append(Variable(t.float()))
                labels.append(Variable(l.float()))

            if torch.cuda.is_available():
                train = [t.cuda() for t in train]
                labels = [l.cuda() for l in labels]

            optimizer.zero_grad()

            regressed_w, mus, logvars = net(w0)  # regressed_w[-1] is the intercept
            

            BCE, KLD = net.vae_loss(regressed_w, w0, mus, logvars)

            log_dic['BCE'] += BCE.data[0]
            log_dic['KLD'] += KLD.data[0]

            total_loss  = BCE + KLD

            total_loss.backward()
            optimizer.step()

            # write to tensorboard
            if global_step % args.write_every_n == 0 and global_step != 0:

                writer.add_scalar('total_loss', 
                        (log_dic['BCE'] + log_dic['KLD'])/args.write_every_n,
                        global_step)
                writer.add_scalar('BCE',
                        log_dic['BCE']/args.write_every_n, global_step)
                writer.add_scalar('KLD',
                        log_dic['KLD']/args.write_every_n, global_step)
                writer.add_scalar('learning_rate',
                        optimizer.state_dict()['param_groups'][0]['lr'],
                        global_step)

                log_dic['BCE'], log_dic['KLD'] = 0, 0

            ########################
            ###### VALIDATION ######
            ########################

            if global_step % args.validate_every_n == 0 and global_step != 0:

                # save computation
                for p in net.parameters():
                    p.requires_grad = False
                
                net.eval()

                val_BCE, val_KLD = 0, 0
                for val_sample in val_dataloader:

                    w0_val = [Variable(x.float()) for x in val_sample['w0']]
                    w1_val = [Variable(x.float()) for x in val_sample['w1']]

                    train_val = []
                    labels_val = []

                    for t, l in val_sample['train']:
                        train_val.append(Variable(t.float()))
                        labels_val.append(Variable(l.float()))

                    if torch.cuda.is_available():
                        w0_val = [x.cuda() for x in w0_val]
                        w1_val = [x.cuda() for x in w1_val]
                        train_val = [t.cuda() for t in train_val]
                        labels_val = [l.cuda() for l in labels_val]

                    regressed_val, mus_val, logvars_val = net(w0_val)

                    BCE_val, KLD_val = net.vae_loss(
                            regressed_val,
                            w0_val,
                            mus_val,
                            logvars_val)

                    val_BCE += BCE_val.data[0]
                    val_KLD += KLD_val.data[0]

                writer.add_scalar('BCE/val', val_BCE/len(val_dataloader),
                        global_step)
                writer.add_scalar('KLD/val', val_KLD/len(val_dataloader),
                        global_step)

                # reset params
                for p in net.parameters():
                    p.requires_grad = True
                net.train()

            if global_step % args.classif_every_n == 0  and global_step != 0:

                net.regressor = True
                scoring.check_performance(net, val_dataloader, writer, args, global_step)
                net.regressor = False

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
    parser.add_argument('--mnist')
    parser.add_argument('--extended', action='store_true')
    parser.add_argument('--val_labels', nargs='+', type=int,
            help='Which labels to use as validation')
    parser.add_argument('--index')
    parser.add_argument('-r','--runs', dest='r')
    parser.add_argument('--ckpt')
    parser.add_argument('--cpu', action='store_true')

    # architecture args
    parser.add_argument('--net',type=str, default='mlp_reg')

    # training args
    parser.add_argument('--optimiser', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--betas', nargs='+', type=float, default=[0.9, 0.999])

    # logging args
    parser.add_argument('--write_every_n', type=int, default=500)
    parser.add_argument('--validate_every_n', type=int, default=5000)
    parser.add_argument('--classif_every_n', type=int, default=10000)
    parser.add_argument('--save_every_n', type=int, default=1)
    parser.add_argument('--n_models_to_keep', type=int, default=3)
    args = parser.parse_args()

    if args.cpu:
        torch.cuda.is_available = lambda: False

    train(args)
