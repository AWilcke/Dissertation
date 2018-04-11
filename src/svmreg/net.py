from torchvision.models import alexnet
import torch
from torch import nn
from torch.utils import model_zoo

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'local': '/home/s1452854/Dissertation/models',
}

def make_alexnet(regressed_w):

    net = alexnet(pretrained=True)

    # replace last layer
    net.classifier.add_module('6', nn.Linear(4096, 1))
    # net.classifier.add_module('7', nn.Sigmoid())
    
    # load weights in
    net.classifier.state_dict()['6.weight'].copy_(regressed_w[:-1])
    net.classifier.state_dict()['6.bias'].copy_(regressed_w[-1:])

    return net
