import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torch
from torch.utils.data.dataloader import default_collate
from torch.autograd import Variable
from collections import defaultdict
from utils import MNISTbyClass
from torch.utils.data import DataLoader


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
    layers = len(batch[0]['w0'])

    out = {}
    for key in ('w0','w1'):
        out[key] = []
        for i in range(layers):
            out[key].append(default_collate([d[key][i] for d in batch]))

    out['label'] = default_collate([d['label'] for d in batch])

    # list of tensors for training samples
    out['train'] = [d['train'] for d in batch]

    return out

def dropout_train(m):
    '''
    Sets dropout modules to train mode, for added noise in the generator
    '''
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        m.train()

def dict_to_tensor_list(state_dict):

    layers = []
    seen = set()
    for key in state_dict.keys():
        name = '.'.join(key.split('.')[:-1])
        if name not in seen:
            seen.add(name)
            layers.append(name)

    del seen
    return [torch.cat(
        [state_dict["{}.weight".format(x)], state_dict["{}.bias".format(x)]], dim=1) \
                for x in layers]

def copy_tensor_list_to_net(tensor_list, net):

    for i, item in enumerate(net.state_dict().values()):
        current = tensor_list[i//2]
        part = current[:,:-1] if i % 2 == 0 else current[:,-1]
        item.copy_(part)

    return net

def validation_metrics(net, val_dataloader, writer, global_step):

    # save computation
    for p in net.parameters():
        p.requires_grad = False
    
    net.eval()
    net.apply(dropout_train)

    val_l2, val_hinge = 0, 0
    for val_sample in val_dataloader:

        w0_val = [Variable(x) for x in val_sample['w0'].float()]
        w1_val = [Variable(x) for x in val_sample['w1'].float()]

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

        regressed_val = net(w0_val)

        val_l2_loss = net.l2_loss(regressed_val, w1_val)
        val_hinge_loss = net.perf_loss(regressed_val, train_val, labels_val)

        val_l2 += val_l2_loss.data[0]
        val_hinge += val_hinge_loss.data[0]

    writer.add_scalar('l2_loss/val', val_l2/len(val_dataloader),
            global_step)
    writer.add_scalar('hinge_loss/val', val_hinge/len(val_dataloader),
            global_step)


    # reset params
    for p in net.parameters():
        p.requires_grad = True
    net.train()

def check_performance(net, val_dataloader, writer, args, global_step):
    # save computation
    for p in net.parameters():
        p.requires_grad = False
    
    net.eval()
    net.apply(dropout_train)

    for val_sample in val_dataloader:

        w0_val = [Variable(x) for x in val_sample['w0'].float()]

        if torch.cuda.is_available():
            w0_val = [x.cuda() for x in w0_val]

        regressed_val = net(w0_val)

        l = net.tensor_dict(regressed_val)

        correct = defaultdict(int)
        total = defaultdict(int)

        for b in range(regressed_val[0].size(0)):
            mnist = MNISTbyClass(args.mnist, args.index, int(val_sample['label'][b]), train_labels=False, train_split=False)
            loader = DataLoader(mnist, batch_size=256, num_workers=0)
            n = val_sample['train'][b][1].size(0)

            for ipt, labels in loader:
                out = net.fprop(l, ipt, b)
                pred = (out.data > 0.5).float()
                correct[n] += (pred==labels.data).sum()
                total[n] += labels.size(0)

    accuracies = [correct[n]/total[n] for num in sorted(correct.keys())]
    im = make_graph_image(np.arange(1,len(accuracies)+1), accuracies)
    latex = ''.join(['({:.0f},{:.4f})'.format(n+1, acc) for n, acc in enumerate(accuracies)])

    writer.add_image('classification', im, global_step)
    writer.add_text('latex', latex, global_step)

    # reset params
    for p in net.parameters():
        p.requires_grad = True
    net.train()
