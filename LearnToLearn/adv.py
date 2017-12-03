import torch
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from dataloader import SVMDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
import argparse
from tensorboardX import SummaryWriter
import os
from models import SVMRegressor, Critic, LargeCritic
import re
from collections import defaultdict
from scoring import score_svm
import pickle
from utils import make_graph_image 
import numpy as np

BATCH_SIZE = 64 # do not set to 1, as BatchNorm won't work
NUM_EPOCHS = 150000

def collate_fn(batch):
    # default collate w0, w1
    out = {key: default_collate([d[key] for d in batch]) for key in ('w0','w1')}

    # list of tensors for training samples
    out['train'] = [d['train'] for d in batch]

    return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def dropout_train(m):
    '''
    Sets dropout modules to train mode, for added noise in the generator
    '''
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        m.train()

def gradient_penalty(critic, real, fake):

    epsilon = torch.rand(real.shape[0], 1)
    # expand to size of real
    epsilon = epsilon.expand(real.shape[0], int(real.nelement()/real.shape[0])).contiguous().view(real.size())
    if isinstance(real, torch.cuda.FloatTensor):
        epsilon = epsilon.cuda()

    xhat = Variable(epsilon * real + (1-epsilon) * fake, requires_grad=True)

    critic_out = critic(xhat)

    ones = torch.ones(critic_out.size())
    if isinstance(real, torch.cuda.FloatTensor):
        ones = ones.cuda()

    grads = autograd.grad(outputs=critic_out, inputs=xhat, grad_outputs=ones,
            create_graph=True, only_inputs=True)[0]

    return (torch.norm(grads) - 1).pow(2)

def train(args):
    # training datasets
    dataset = SVMDataset(args.w0, args.w1, args.feature, split='train')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
            shuffle=True, num_workers=8, collate_fn=collate_fn)
    # validation datasets
    val_dataset = SVMDataset(args.w0, args.w1, args.feature, split='val')
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
            shuffle=False, num_workers=8, collate_fn=collate_fn)
    
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
    
    net = SVMRegressor(
            slope=args.slope,
            dropout=args.dropout,
            square_hinge=args.square_hinge, 
            n_gpu=n_gpu)
    net.apply(weights_init)
    if args.large_critic:
        critic = LargeCritic(n_gpu=n_gpu, gp=args.gp)
    else:
        critic = Critic(n_gpu=n_gpu, gp=args.gp)
    critic.apply(weights_init)

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

    if args.optimiser == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr_G, betas=(0, 0.9))
        optimizer_c = optim.Adam(critic.parameters(), lr=args.lr_C, betas=(0, 0.9))
    elif args.optimiser == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr_G)
        optimizer_c = optim.RMSprop(critic.parameters(), lr=args.lr_C)
    elif args.optimiser == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr_G, momentum=args.momentum)
        optimizer_c = optim.SGD(critic.parameters(), lr=args.lr_C, momentum=args.momentum)
    else:
        raise Exception("Optimiser type unkown : {}".format(args.optimiser))

    print("Using {} optimiser".format(args.optimiser))

    running_c, running_g, running_hinge, running_grad, running_l2 = 0, 0, 0, 0, 0
    gen_counter, critic_counter = 0, 0

    one = torch.FloatTensor([1])
    mone = -1 * one

    if torch.cuda.is_available():
        one, mone = one.cuda(), mone.cuda()

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

                if not args.gp:
                    for p in critic.parameters():
                        p.data.clamp_(-0.01, 0.01)
                
                w0 = Variable(samples['w0'].float(), volatile=True)
                w1 = Variable(samples['w1'].float())

                if torch.cuda.is_available():
                    w0 = w0.cuda()
                    w1 = w1.cuda()

                optimizer_c.zero_grad()

                # train with real (w1)
                errC_real = torch.mean(critic(w1))
                errC_real.backward(mone)

                # train with fake
                fake_w1 = Variable(net(w0).data)
                errC_fake = torch.mean(critic(fake_w1))
                errC_fake.backward(one)

                # apply gradient penalty
                if args.gp:
                    grad_penalty = gradient_penalty(critic, w1.data, fake_w1.data)
                    grad_penalty.backward(args.lambd * one)

                    running_grad += grad_penalty.data[0]

                optimizer_c.step()

                err_C = errC_real - errC_fake
                running_c += err_C.data[0]
                critic_counter += 1

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
                err_G = torch.mean(critic_out)


                # train with hinge loss
                l2_loss, hinge_loss = net.loss(regressed_w, w1, train)

                # training a pure gan
                if args.pure_gan:
                    total_loss = - err_G
                else:
                    total_loss = args.alpha * -err_G + args.delta * (hinge_loss + l2_loss)
                total_loss.backward()

                optimizer.step()

                running_g += err_G.data[0]
                running_hinge += hinge_loss.data[0]
                running_l2 += l2_loss.data[0]

                gen_iterations += 1
                gen_counter += 1

            ########################
            ####### LOGGING ########
            ########################

            print('[%d/%d][%d] Loss_C: %.3f Loss_G: %.3f Loss_C_real: %.3f Loss_C_fake: %.3f Loss_Hinge: %.3f Loss_l2: %.3f'
            % (i, len(dataloader), gen_iterations,
            err_C.data[0], err_G.data[0], errC_real.data[0], errC_fake.data[0], hinge_loss.data[0], l2_loss.data[0]))

            # write to tensorboard
            if gen_iterations % args.write_every_n == 0:
                # make sure we're not writing the tail end of a batch
                if gen_counter > 5:
                    writer.add_scalar('err_G', 
                            running_g/gen_counter, gen_iterations)
                    writer.add_scalar('hinge_loss', 
                            running_hinge/gen_counter, gen_iterations)
                    writer.add_scalar('l2_loss',
                            running_l2/gen_counter, gen_iterations)
                if critic_counter > 5:
                    writer.add_scalar('err_D',
                            running_c/critic_counter, gen_iterations)
                    if args.gp:
                        writer.add_scalar('grad',
                                running_grad/critic_counter, gen_iterations)
                writer.add_scalar('learning_rate',
                        optimizer.state_dict()['param_groups'][0]['lr'],
                        gen_iterations)

                # reset all counters
                running_g = 0
                running_c = 0
                running_hinge = 0
                running_l2 = 0
                running_grad = 0
                gen_counter = 0
                critic_counter = 0

            # save model
            if gen_iterations % args.save_every_n == 0:

                torch.save(net.state_dict(),
                        os.path.join(args.ckpt, "{}.ckpt".format(gen_iterations))
                        )
                torch.save(critic.state_dict(),
                        os.path.join(args.ckpt, "{}_critic.ckpt".format(gen_iterations)))

                # remove old models
                models = [f for f in os.listdir(args.ckpt) if ".ckpt" in f]
                models = sorted(models,
                        key=lambda x : int(re.findall(r'\d+', x)[0]),
                        reverse=True)

                while len(models) > 2 * args.n_models_to_keep:
                    os.remove(os.path.join(args.ckpt, models.pop()))

            ########################
            ###### VALIDATION ######
            ########################

            # get validation metrics for G/C
            
            if gen_iterations % args.validate_every_n == 0:

                # save computation
                for p in net.parameters():
                    p.requires_grad = False
                
                net.eval()
                net.apply(dropout_train)

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
                        gen_iterations)
                writer.add_scalar('hinge_loss/val', val_hinge/len(val_dataloader),
                        gen_iterations)


                # reset params
                for p in net.parameters():
                    p.requires_grad = True
                net.train()

            if gen_iterations % args.classif_every_n == 0:

                # save computation
                for p in net.parameters():
                    p.requires_grad = False
                
                net.eval()
                net.apply(dropout_train)
            
                # get regressed classification
                x = val_dataset.features

                svm_params = {'loss' : 'square_hinge', 'dual':False}
                scores = defaultdict(list)
                for sample in (os.path.join(args.w0, 'val', w) \
                        for w in os.listdir(os.path.join(args.w0, 'val'))):
                    with open(sample, 'rb') as f:
                        s = pickle.load(f)

                    n, acc = score_svm(s, x, y, net, svm_params=svm_params)

                    scores[n].append(acc)

                accuracies = [np.mean(scores[num]) for num in sorted(scores.keys())]
                im = make_graph_image(np.arange(len(accuracies)), accuracies)
                t = transforms.ToTensor()

                writer.add_image('classification', t(im), gen_iterations)

                # reset params
                for p in net.parameters():
                    p.requires_grad = True
                net.train()

if __name__ == "__main__":
    
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
    parser.add_argument('--optimiser', type=str, default='sgd')
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
    parser.add_argument('--square_hinge', action='store_true')
    parser.add_argument('--pure_gan', action='store_true')
    parser.add_argument('--gp', action='store_true')
    parser.add_argument('--large_critic', action='store_true')

    # logging args
    parser.add_argument('--write_every_n', type=int, default=100)
    parser.add_argument('--validate_every_n', type=int, default=500)
    parser.add_argument('--classif_every_n', type=int, default=10000)
    parser.add_argument('--save_every_n', type=int, default=1000)
    parser.add_argument('--n_models_to_keep', type=int, default=1)
    args = parser.parse_args()

    train(args)
