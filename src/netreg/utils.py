import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torch
from torch.utils.data.dataloader import default_collate

def weight_init(m):

    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        print(classname)
        dim = m.weight.data.size(0)
        m.weight.data.copy_(torch.eye(dim))
        m.bias.data.fill_(0)

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
    # returns conv as Nout * (Nin*(f*f+b))
    # TODO might want to transpose Ns?

    layers = []
    seen = set()
    for key in state_dict.keys():
        name = '.'.join(key.split('.')[:-1])
        if name not in seen:
            seen.add(name)
            layers.append(name)

    del seen

    out = []
    for x in layers:
        weights = state_dict[f"{x}.weight"]
        bias = state_dict[f"{x}.bias"]
        out.append(torch.cat([weights.view(weights.size(0), -1), bias.view(-1, 1)], dim=1))

    return out

def copy_tensor_list_to_net(tensor_list, net):

    for i, item in enumerate(net.state_dict().values()):
        current = tensor_list[i//2]
        part = current[:,:-1] if i % 2 == 0 else current[:,-1]
        item.copy_(part)

    return net
