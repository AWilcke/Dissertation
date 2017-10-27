import torch
import torch.optim as optim
from torch.autograd import Variable
from dataloader import SVMDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import argparse
from tensorboardX import SummaryWriter
import os
from models import SVMRegressor, Critic

BATCH_SIZE = 64 # do not set to 1, as BatchNorm won't work
NUM_EPOCHS = 100000

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

    writer = SummaryWriter(args.r)

    if not os.path.exists(args.ckpt):
        os.mkdir(args.ckpt)
    
    net = SVMRegressor(square_hinge=args.square_hinge)
    critic = Critic()

    if torch.cuda.is_available():
        net = net.cuda()
        critic = critic.cuda()

    if args.optimiser == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
        optimizer_c = optim.Adam(critic.parameters(), lr=args.lr)
    elif args.optimiser == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr)
        optimizer_c = optim.RMSprop(critic.parameters(), lr=args.lr)
    elif args.optimiser == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
        optimizer_c = optim.SGD(critic.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        raise Exception("Optimiser type unkown : {}".format(args.optimiser))

    gen_iterations = 0
    running_c, running_g = 0, 0
    gen_counter, critic_counter = 0, 0

    while gen_iterations < NUM_EPOCHS:

        data_iter = iter(dataloader)
        i = 0

        while i < len(dataloader):

            #####################
            ### Update Critic ###
            #####################

            for p in critic.parameters():
                p.requires_grad = True

            if gen_iterations < 25 or gen_iterations % 500 == 0:
                critic_iterations = 100
            else:
                critic_iterations = 5

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

                # train with real (w1)
                errC_real = torch.mean(critic(w1))

                # train with fake
                fake_w1 = Variable(net(w0).data)
                errC_fake = torch.mean(critic(fake_w1))

                err_C = errC_real - errC_fake
                err_C.backward()
                optimizer_c.step()

                for p in critic.parameters():
                    p.data.clamp_(-0.01, 0.01)

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

                if torch.cuda.is_available():
                    w0 = w0.cuda()
                    w1 = w1.cuda()

                optimizer.zero_grad()
                
                regressed_w = net(w0) # regressed_w[-1] is the intercept

                critic_out = critic(regressed_w)
                err_G = torch.mean(critic_out)
                err_G.backward()
                optimizer.step()

                running_g += err_G.data[0]
                gen_iterations += 1
                gen_counter += 1

            # write to tensorboard
            if gen_iterations % args.write_every_n == 0:

                writer.add_scalar('err_G', 
                        running_g/gen_counter, gen_iterations)
                writer.add_scalar('err_D',
                        running_c/critic_counter, gen_iterations)
                writer.add_scalar('learning_rate',
                        optimizer.state_dict()['param_groups'][0]['lr'],
                        gen_iterations)

                running_g = 0
                running_c = 0
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
                models = [os.path.join(args.ckpt, f) for 
                        f in os.listdir(args.ckpt) if ".ckpt" in f]
                models = sorted(models,
                        key=lambda x : int(os.path.basename(x)[0]),
                        reverse=True)

                while len(models) > 2 * args.n_models_to_keep:
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
    parser.add_argument('--optimiser', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr_weight', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=5e-2)
    parser.add_argument('--steps', nargs='+', type=int, default=[3,6])
    parser.add_argument('--step_gamma', type=float, default=0.1)
    parser.add_argument('--square_hinge', action='store_true')

    # logging args
    parser.add_argument('--write_every_n', type=int, default=100)
    parser.add_argument('--validate_every_n', type=int, default=5)
    parser.add_argument('--save_every_n', type=int, default=1000)
    parser.add_argument('--n_models_to_keep', type=int, default=1)
    args = parser.parse_args()

    train(args)
