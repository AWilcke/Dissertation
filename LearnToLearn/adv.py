import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from dataloader import SVMDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import argparse
from tensorboardX import SummaryWriter
import os
from models import SVMRegressor, Critic

BATCH_SIZE = 16 # do not set to 1, as BatchNorm won't work
NUM_EPOCHS = 100

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
    
    net = SVMRegressor(square_hinge=args.square_hinge)
    critic = Critic()
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

    # # lr step
    # lr_schedule = lr_scheduler.MultiStepLR(optimizer, args.steps, gamma=args.step_gamma)

    # #update optimiser, in case loading model
    # for i in range(start_epoch):
        # lr_schedule.step()

    gen_iterations = 0

    for epoch in range(start_epoch, NUM_EPOCHS):

        running_critic, running_hinge = 0, 0
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

                for p in critic.parameters():
                    p.data.clamp(-0.01, 0.01)

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
                
                _ , hinge_loss = net.loss(regressed_w, w1, train)
                critic_out = critic(regressed_w)
                critic_loss = torch.mean(critic_out)
                running_critic += critic_loss.data[0]
                running_hinge += hinge_loss.data[0]

                # total_loss  = l2_loss + args.lr_weight * hinge_loss + critic_loss
                total_loss = args.lr_weight * hinge_loss + critic_loss

                total_loss.backward()
                optimizer.step()
                gen_iterations += 1

            global_step = gen_iterations       

            # write to tensorboard
            if global_step % args.write_every_n == 0:

                writer.add_scalar('total_loss/train', 
                        (running_critic + running_hinge)/args.write_every_n,
                        global_step)
                # writer.add_scalar('l2_loss/train',
                        # running_l2/args.write_every_n, global_step)
                writer.add_scalar('critic_loss/train',
                        running_critic/args.write_every_n, global_step)
                writer.add_scalar('hinge_loss/train',
                        running_hinge/args.write_every_n, global_step)
                writer.add_scalar('learning_rate',
                        optimizer.state_dict()['param_groups'][0]['lr'],
                        global_step)

                running_critic, running_hinge = 0, 0

            # run validation cycle
            if global_step % args.validate_every_n == 0:
                
                # clear up space on gpu
                del w0, w1, train
                
                net.train(False)

                val_critic, val_hinge = 0, 0
                for val_sample in val_dataloader:

                    w0_val = Variable(val_sample['w0'].float())
                    w1_val = Variable(val_sample['w1'].float())
                    train_val = [Variable(t.float()) for t in val_sample['train']]

                    if torch.cuda.is_available():
                        w0_val = w0_val.cuda()
                        w1_val = w1_val.cuda()
                        train_val = [t.cuda() for t in train_val]

                    regressed_val = net(w0_val)

                    _, val_hinge_loss = net.loss(regressed_val, w1_val, train_val)
                    val_critic_loss = critic(regressed_val)

                    val_critic += val_critic_loss.data[0]
                    val_hinge += val_hinge_loss.data[0]

                writer.add_scalar('critic_loss/val', val_critic/len(val_dataloader),
                        global_step)
                writer.add_scalar('hinge_loss/val', val_hinge/len(val_dataloader),
                        global_step)
                writer.add_scalar('total_loss/val', 
                        (val_critic + val_hinge)/len(val_dataloader),
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
    parser.add_argument('--optimiser', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lr_weight', type=float, default=1)
    parser.add_argument('--steps', nargs='+', type=int, default=[3,6])
    parser.add_argument('--step_gamma', type=float, default=0.1)
    parser.add_argument('--square_hinge', action='store_true')

    # logging args
    parser.add_argument('--write_every_n', type=int, default=10)
    parser.add_argument('--validate_every_n', type=int, default=50)
    parser.add_argument('--save_every_n', type=int, default=1)
    parser.add_argument('--n_models_to_keep', type=int, default=3)
    args = parser.parse_args()

    train(args)
