from torchvision.models import alexnet
import torch
from torch import nn
from torch.utils import model_zoo
from sklearn.svm import LinearSVC
import numpy as np

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'local': '/home/s1452854/Dissertation/models',
}

def refit_svm(regressed_w, correct_i, wrong_i, data):
    svm = LinearSVC(regressed_w=regressed_w, dual=False)
    shuffle = np.random.permutation(2*len(correct_i))
    y = np.array([1] * len(correct_i) + [0] * len(wrong_i))
    index = np.concatenate((correct_i, wrong_i))
    svm.fit(data[index][shuffle], y[shuffle])
    return np.concatenate(
            (svm.coef_,
                svm.intercept_.reshape(1,1)),
            axis=1)[0]

def make_alexnet(regressed_w):

    net = alexnet(pretrained=True)

    # replace last layer
    net.classifier.add_module('6', nn.Linear(4096, 1))
    # net.classifier.add_module('7', nn.Sigmoid())
    
    # load weights in
    net.classifier.state_dict()['6.weight'].copy_(regressed_w[:-1])
    net.classifier.state_dict()['6.bias'].copy_(regressed_w[-1:])

    return net
