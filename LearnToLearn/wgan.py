import torch
import torch.optim as optim
from torch.autograd import Variable
from dataloader import SVMDataset
from torch.utils.data import DataLoader
import argparse
from tensorboardX import SummaryWriter
import os
from models import SVMRegressor, Generator8, UGen, Critic3, Critic4, Critic8
import re
import pickle
import utils
from utils import collate_fn, weights_init
import resource
import losses

BATCH_SIZE = 64 # do not set to 1, as BatchNorm won't work
NUM_EPOCHS = 150000

model_dict = {
        'standard':SVMRegressor,
        'gen8':Generator8,
        'ugen':UGen,
        'critic3':Critic3,
        'critic4':Critic4,
        'critic8':Critic8
        }

def train(args):
    # training datasets
    dataset = SVMDataset(args.w0, args.w1, args.feature, split='train')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
            shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=True)
    # validation datasets
    val_dataset = SVMDataset(args.w0, args.w1, args.feature, split='val')
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
            shuffle=False, num_workers=0, collate_fn=collate_fn, drop_last=True)
    
    # store labels for validation
    with open(args.labels,'rb') as fi:
        y = pickle.load(fi)

    writer = SummaryWriter(args.r)
    
    # create models
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print("Using {} GPUs".format(n_gpu))
    else:
        n_gpu = 0
    
    
    net = model_dict[args.gen_name](
            slope=args.slope,
            dropout=args.dropout,
            square_hinge=args.square_hinge, 
            n_gpu=n_gpu)
    net.apply(weights_init)

    critic = model_dict[args.critic_name](n_gpu=n_gpu, gp=args.gp, dropout=args.no_dropout_C)
    critic.apply(weights_init)

    if args.type == 'wgan':
        gradient_penalty = losses.wgan_gradient_penalty
        gen_loss = losses.wgan_gen_loss
        critic_loss = losses.wgan_critic_loss
        extra = Variable(None) # no extra variable needed for wgan
    elif args.type == 'dragan':
        gradient_penalty = losses.dragan_gradient_penalty
        gen_loss = losses.dragan_gen_loss
        critic_loss = losses.dragan_critic_loss
        extra = Variable(torch.FloatTensor(BATCH_SIZE,1))
    elif args.type == 'fisher':
        if args.gp:
            raise Exception('FisherGAN does not support gradient penalty')
        gen_loss = losses.wgan_gen_loss # fisher uses same gen_loss as wgan
        critic_loss = losses.fisher_critic_loss
        extra = Variable(torch.FloatTensor([0]), requires_grad=True) # lagrange multipliers

    print(net)
    print(critic)

    gen_iterations = 0
    
    # restore from ckpt
    if not os.path.exists(args.ckpt):
        os.mkdir(args.ckpt)
    if os.listdir(args.ckpt):
        # get all models
        models = [f  for f in os.listdir(args.ckpt) if ".ckpt" in f]
        # sort by gen_iter
        latest_models = sorted(models,
                key=lambda x : int(re.findall(r'\d+', x)[0]),
                reverse=True)[:2]
        # get critic and gen
        latest_gen, latest_critic = sorted(latest_models)

        print("Restoring from model {}".format(latest_gen))
        net.load_state_dict(torch.load(
            os.path.join(args.ckpt, latest_gen)))
        print("Restoring from model {}".format(latest_critic))
        critic.load_state_dict(torch.load(
            os.path.join(args.ckpt, latest_critic)))
        
        #update start iter
        gen_iterations = int(latest_gen.split('.')[0])


    if torch.cuda.is_available():
        net = net.cuda()
        critic = critic.cuda()

    if args.optimiser_C == 'adam':
        optimizer_c = optim.Adam(critic.parameters(), lr=args.lr_C, betas=tuple(args.betas))
    elif args.optimiser_C == 'rmsprop':
        optimizer_c = optim.RMSprop(critic.parameters(), lr=args.lr_C)
    elif args.optimiser_C == 'sgd':
        optimizer_c = optim.SGD(critic.parameters(), lr=args.lr_C, momentum=args.momentum)
    else:
        raise Exception("Optimiser type unkown : {}".format(args.optimiser_C))

    if args.optimiser_G == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr_G, betas=tuple(args.betas))
    elif args.optimiser_G == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr_G)
    elif args.optimiser_G == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr_G, momentum=args.momentum)
    else:
        raise Exception("Optimiser type unkown : {}".format(args.optimiser_G))

    print("Using {}, {} optimisers".format(args.optimiser_C, args.optimiser_G))

    log_dic = {
            'C':0,
            'G':0,
            'hinge':0,
            'grad':0,
            'l2':0,
            'C_count':0,
            'G_count':0,
            }
    

    one = torch.FloatTensor([1])
    mone = -1 * one

    if torch.cuda.is_available():
        one, mone = one.cuda(), mone.cuda()
        extra = extra.cuda()

    while gen_iterations < NUM_EPOCHS:

        data_iter = iter(dataloader)
        i = 0

        while i < len(dataloader):

            #####################
            ### Update Critic ###
            #####################

            for p in critic.parameters():
                p.requires_grad = True

            if gen_iterations < 25:
                critic_iterations = 100
            else:
                critic_iterations = args.critic_iters

            j = 0
            while j < critic_iterations and i < len(dataloader):
                j += 1
                samples = data_iter.next()
                i+=1
                
                w0 = Variable(samples['w0'].float(), volatile=True)
                w1 = Variable(samples['w1'].float())

                if torch.cuda.is_available():
                    w0 = w0.cuda()
                    w1 = w1.cuda()

                optimizer_c.zero_grad()

                fake_w1 = Variable(net(w0).data)

                err_C = critic_loss(critic, w1, fake_w1, one, mone, args, extra)

                # apply gradient penalty
                if args.gp:
                    grad_penalty = gradient_penalty(critic, w1.data, fake_w1.data)
                    grad_penalty.backward(args.lambd * one)

                    log_dic['grad'] += grad_penalty.data[0]

                optimizer_c.step()

                log_dic['C'] += err_C.data[0]
                log_dic['C_count'] += 1

            ########################
            ### Update Regressor ###
            ########################

            for p in critic.parameters():
                p.requires_grad = False
            
            if i < len(dataloader):

                samples = data_iter.next()
                i+=1

                w0 = Variable(samples['w0'].float())
                w1 = Variable(samples['w1'].float())
                train = [Variable(t.float()) for t in samples['train']]

                if torch.cuda.is_available():
                    w0 = w0.cuda()
                    w1 = w1.cuda()
                    train = [t.cuda() for t in train]

                optimizer.zero_grad()
                
                regressed_w = net(w0) # regressed_w[-1] is the intercept

                # train with critic loss
                critic_out = critic(regressed_w)

                err_G = gen_loss(critic_out)

                # train with hinge loss
                l2_loss, hinge_loss = net.loss(regressed_w, w1, train)

                # training a pure gan
                if args.pure_gan:
                    total_loss = - err_G
                else:
                    total_loss = args.alpha * -err_G + args.delta * (hinge_loss + l2_loss)

                total_loss.backward()

                optimizer.step()

                log_dic['G'] += err_G.data[0]
                log_dic['hinge'] += hinge_loss.data[0]
                log_dic['l2'] += l2_loss.data[0]

                gen_iterations += 1
                log_dic['G_count'] += 1

            ########################
            ####### LOGGING ########
            ########################

            # write to tensorboard
            if gen_iterations % args.write_every_n == 0:
                log_dic = utils.log_to_tensorboard(log_dic,
                    optimizer.state_dict()['param_groups'][0]['lr'],
                    args.gp,
                    writer,
                    gen_iterations)


            # save model
            if gen_iterations % args.save_every_n == 0:

                utils.save_model(net, critic, args, gen_iterations)

            ########################
            ###### VALIDATION ######
            ########################

            # get validation metrics for G/C
            
            if gen_iterations % args.validate_every_n == 0:

                utils.validation_metrics(net, val_dataloader, writer, gen_iterations)

            if gen_iterations % args.classif_every_n == 0:

                utils.check_performance(net, val_dataset, y, writer, args, gen_iterations)

if __name__ == "__main__":
    
    # trying to prevent ancdata error
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    
    parser = argparse.ArgumentParser()
    # io args
    parser.add_argument('-w0')
    parser.add_argument('-w1')
    parser.add_argument('-f', dest='feature')
    parser.add_argument('-l', dest='labels')
    parser.add_argument('-r','--runs', dest='r')
    parser.add_argument('--ckpt')

    # training args
    parser.add_argument('--critic_iters', type=int, default=5)
    parser.add_argument('--optimiser_C', type=str, default='sgd')
    parser.add_argument('--optimiser_G', type=str, default='adam')
    parser.add_argument('--betas', nargs='+', type=float, default=[0, 0.9])
    parser.add_argument('--lr_C', type=float, default=5e-5)
    parser.add_argument('--lr_G', type=float, default=5e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--steps', nargs='+', type=int, default=[3,6])
    parser.add_argument('--step_gamma', type=float, default=0.1)
    
    # balancing/architecture args
    parser.add_argument('--alpha', type=float, default=1,
            help='adverserial loss weight')
    parser.add_argument('--delta', type=float, default=1,
            help='hinge+l2 loss weight')
    parser.add_argument('--lambda', dest='lambd', type=float, default=10,
            help='gradient penalty weight for wgan-gp')
    parser.add_argument('--slope', type=float, default=0.01,
            help='negative slope for LeakyRelu')
    parser.add_argument('--dropout', type=float, default=0,
            help='dropout probability for regressor')
    parser.add_argument('--no_dropout_C', action='store_false',
            help='whether to omit dropout from critic')
    parser.add_argument('--square_hinge', action='store_true')
    parser.add_argument('--pure_gan', action='store_true')
    parser.add_argument('--gp', action='store_true')
    parser.add_argument('--gen_name',type=str, default='standard')
    parser.add_argument('--critic_name',type=str, default='critic3')

    # logging args
    parser.add_argument('--write_every_n', type=int, default=100)
    parser.add_argument('--validate_every_n', type=int, default=500)
    parser.add_argument('--classif_every_n', type=int, default=10000)
    parser.add_argument('--save_every_n', type=int, default=1000)
    parser.add_argument('--n_models_to_keep', type=int, default=1)
    args = parser.parse_args()

    train(args)
