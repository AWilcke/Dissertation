from torchvision.models import alexnet
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader
from dataloader import BasicDataset
from torch.autograd import Variable
from torchvision import transforms, datasets
import argparse
import numpy as np
import pickle
import os
import progressbar

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'local': '../../models/',
}

BATCH_SIZE = 256

def get_feature_extractor():
    """
    Downloads a pretrained AlexNet and removes the last fully connected layer,
    creating deep features that can be used for other tasks.
    """
    net = alexnet(pretrained=False)
    net.load_state_dict(model_zoo.load_url(model_urls['alexnet'], 
        model_dir=model_urls['local']))

    feature_extractor = nn.Sequential(*list(net.classifier.children())[:-1])
    net.classifier = feature_extractor
    return net

def main(args):
    """
    Creates dataset from linked labels file and root directory, then extracts
    features for all images.
    """
    data_transform = transforms.Compose([
        transforms.Scale((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])
    dataset = datasets.ImageFolder(root=args.root_dir, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=BATCH_SIZE)
    net = get_feature_extractor()

    if torch.cuda.is_available():
        net = net.cuda()

    features_out = np.zeros((len(dataset), 4096))
    labels_out = np.zeros(len(dataset))
    
    p = progressbar.ProgressBar(widgets=[progressbar.ETA(), ' ', progressbar.Percentage()])
    for i, samples in p(enumerate(dataloader)):
        images, labels = samples
        if torch.cuda.is_available():
            images = images.cuda()
        images = Variable(images)
        features = net(images).cpu().data.numpy()
        features_out[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE] = features
        labels_out[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE] = labels.int().numpy()
        print(i)

    with open(os.path.join(args.out, 'features.pickle'),'wb') as f:
        pickle.dump(features_out, f)
    with open(os.path.join(args.out, 'labels.pickle'),'wb') as f:
        pickle.dump(labels_out, f)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--root_dir', dest='root_dir')
    parser.add_argument('-o','--out', dest='out')
    args = parser.parse_args()

    main(args)
