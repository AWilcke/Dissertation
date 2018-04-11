#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# Adapted: Arthur Wilcke
# URL:      http://kazuto1011.github.io
# Created:  2017-05-26

from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import transforms
import argparse
import pickle
from net import make_alexnet, refit_svm
import cv2
from PIL import Image
from models import SVMRegressor

parser = argparse.ArgumentParser()
parser.add_argument('--image-path', type=str, default='data/jpg/deli/deli_09_05_altavista.jpg',
        help='Input image path')
parser.add_argument('--target-layer', type=str, default='features.11')
parser.add_argument('--label', type=int)
parser.add_argument('--data-path', default='data/features.pickle')
parser.add_argument('--refit', action='store_true')

parser.add_argument('--model_path')
parser.add_argument('--regressor_path')
parser.add_argument('--adv_path')
parser.add_argument('--w1_path')

parser.add_argument('--save')


class _PropagationBase(object):

    def __init__(self, model):
        super(_PropagationBase, self).__init__()
        self.cuda = True if next(model.parameters()).is_cuda else False
        self.model = model
        self.model.eval()
        self.image = None

    def forward(self, image):
        self.image = image
        self.model.zero_grad()
        self.preds = self.model(self.image)
        self.prob = F.sigmoid(self.preds)[0]
        return self.prob

    def backward(self):
        self.preds.backward(retain_graph=True)


class BackPropagation(_PropagationBase):

    def generate(self):
        output = self.image.grad.data.cpu().numpy()[0]
        return output.transpose(1, 2, 0)


class GuidedBackPropagation(BackPropagation):

    def __init__(self, model):
        super(GuidedBackPropagation, self).__init__(model)

        def func_b(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        for module in self.model.named_modules():
            module[1].register_backward_hook(func_b)


class GradCAM(_PropagationBase):

    def __init__(self, model):
        super(GradCAM, self).__init__(model)
        self.all_fmaps = OrderedDict()
        self.all_grads = OrderedDict()

        def func_f(module, input, output):
            self.all_fmaps[id(module)] = output.data.cpu()

        def func_b(module, grad_in, grad_out):
            self.all_grads[id(module)] = grad_out[0].cpu()

        for module in self.model.named_modules():
            module[1].register_forward_hook(func_f)
            module[1].register_backward_hook(func_b)

    def _find(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm.data[0]

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        return F.adaptive_avg_pool2d(grads, 1)

    def generate(self, target_layer):
        fmaps = self._find(self.all_fmaps, target_layer)
        grads = self._find(self.all_grads, target_layer)
        weights = self._compute_grad_weights(grads)

        gcam = (fmaps[0] * weights[0].data).sum(dim=0)
        gcam = torch.clamp(gcam, min=0.)

        gcam -= gcam.min()
        gcam /= gcam.max()

        return gcam.cpu().numpy()

def save_gradcam(filename, gcam, raw_image):
    h, w = raw_image.size
    gcam = cv2.resize(gcam, (h, w))
    gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
    gcam = gcam.astype(np.float) + np.array(raw_image, dtype=np.float)
    gcam = gcam / gcam.max() * 255.0
    cv2.imwrite(filename, np.uint8(gcam))

def save_gradient(filename, data):
    data -= data.min()
    data /= data.max()
    data *= 255.0
    cv2.imwrite(filename, np.uint8(data))

def preprocess_image(img):
        means=[0.485, 0.456, 0.406]
        stds=[0.229, 0.224, 0.225]

        preprocessed_img = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=means,
                std=stds)
            ])(img)


        preprocessed_img.unsqueeze_(0)
        if torch.cuda.is_available():
            preprocessed_img = preprocessed_img.cuda()
        var = Variable(preprocessed_img, requires_grad=True)
        return var


def run(net, args, name):
    raw_img = Image.open(args.image_path)
    img = preprocess_image(raw_img)

    gcam = GradCAM(net)

    prob = gcam.forward(img)

    print(f'Probability: {prob.data[0]*100:.1f}%')

    gcam.backward()
    region = gcam.generate(args.target_layer)
    save_gradcam(f'{args.save}_{name}_gcam.png', region, raw_img)

    gbp = GuidedBackPropagation(model=net)
    prob = gbp.forward(img)
    gbp.backward()
    feature = gbp.generate()

    h, w = raw_img.size
    region = cv2.resize(region, (h, w))[..., np.newaxis]
    feature = cv2.resize(feature, (h, w))

    output = feature * region

    save_gradient(f'{args.save}_{name}_gcpcam.png', output)


if __name__ == '__main__':

    args = parser.parse_args()

    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    #### LOAD NET ####
    with open(args.model_path, 'rb') as f:
        d = pickle.load(f)
        w0 = torch.from_numpy(d['w0']).float()
        correct_i = d['correct_i']
        wrong_i = d['wrong_i']

    num_images = len(wrong_i)

    net = make_alexnet(w0)

    print(f'Trained with {num_images} images')

    if torch.cuda.is_available():
        net = net.cuda()

    run(net, args, 'original')

    #### REGRESS NET ####
    # init regressor
    regressor = SVMRegressor()
    regressor.eval()

    regressed_net = None
    if args.regressor_path:
        # print(regressor)
        regressor.load_state_dict(
                torch.load(args.regressor_path))

        # run through regression
        w0 = Variable(w0.unsqueeze(0))
        if torch.cuda.is_available():
            regressor = regressor.cuda()
            w0 = w0.cuda()

        w1 = regressor(w0).data
        w1.squeeze_(0)

        # copy to new identical net
        regressed_net = make_alexnet(w1)

        if torch.cuda.is_available():
            regressed_net = regressed_net.cuda()

        run(regressed_net, args, 'regressed')

        if args.refit:
            w1 = refit_svm(w1.cpu().numpy(),
                    correct_i, 
                    wrong_i,
                    data)
            w1 = torch.from_numpy(w1)
            regressed_net = make_alexnet(w1)

            if torch.cuda.is_available():
                regressed_net = regressed_net.cuda()

            run(regressed_net, args, 'regressed_refit')

    #### REGRESS ADV NET ####
    regressed_adv = None
    if args.adv_path:
        regressor.load_state_dict(
                torch.load(args.adv_path))

        # run through regression
        w1 = regressor(w0).data
        w1.squeeze_(0)

        # copy to new identical net
        regressed_adv = make_alexnet(w1)

        if torch.cuda.is_available():
            regressed_adv = regressed_adv.cuda()

        run(regressed_adv, args, 'adv')

        if args.refit:
            w1 = refit_svm(w1.cpu().numpy(),
                    correct_i, 
                    wrong_i,
                    data)
            w1 = torch.from_numpy(w1)
            regressed_adv = make_alexnet(w1)

            if torch.cuda.is_available():
                regressed_adv = regressed_net.cuda()

            run(regressed_adv, args, 'adv_refit')

    #### W1 NET ####
    w1_net = None
    if args.w1_path:
        with open(args.w1_path, 'rb') as f:
            w1 = torch.from_numpy(pickle.load(f)[args.label]).float()
        w1_net = make_alexnet(w1)

        if torch.cuda.is_available():
            w1_net = w1_net.cuda()

        run(w1_net, args, 'w1')
