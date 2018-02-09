import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
import torch
from torch.utils.data.dataloader import default_collate
from collections import defaultdict
import os
import pickle
from scoring import score_svm
import re


def make_graph_image(x, y):
    plt.switch_backend('agg')
    fig = plt.figure()
    plt.plot(x,y)
    plt.xticks(x)
    fig.canvas.draw()

    w, h  = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(h,w,3)
    t = transforms.ToTensor()

    return t(buf)

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

def validation_metrics(net, val_dataloader, writer, gen_iterations):

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

def check_performance(net, val_dataset, y, writer, args, gen_iterations):
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
    latex = ''.join(['({:.0f},{:.4f})'.format(n+1, acc) for n, acc in enumerate(accuracies)])

    writer.add_image('classification', im, gen_iterations)
    writer.add_text('latex', latex, gen_iterations)

    # reset params
    for p in net.parameters():
        p.requires_grad = True
    net.train()

def save_model(net, critic, args, gen_iterations):
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

def log_to_tensorboard(log_dic, lr, write_gradient, writer, gen_iterations):

    # make sure we're not writing the tail end of a batch
    gen_counter = log_dic['G_count']
    if gen_counter > 5:
        writer.add_scalar('err_G', 
                log_dic['G']/gen_counter, gen_iterations)
        writer.add_scalar('hinge_loss', 
                log_dic['hinge']/gen_counter, gen_iterations)
        writer.add_scalar('l2_loss',
                log_dic['l2']/gen_counter, gen_iterations)

    critic_counter = log_dic['C_count']
    if critic_counter > 5:
        writer.add_scalar('err_D',
                log_dic['C']/critic_counter, gen_iterations)
        if write_gradient:
            writer.add_scalar('grad',
                    log_dic['grad']/critic_counter, gen_iterations)
    writer.add_scalar('learning_rate', lr,
            gen_iterations)

    # reset all counters
    log_dic['G'] = 0
    log_dic['C'] = 0
    log_dic['hinge'] = 0
    log_dic['l2'] = 0
    log_dic['grad'] = 0
    log_dic['G_count'] = 0
    log_dic['C_count'] = 0

    return log_dic
