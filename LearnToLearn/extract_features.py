from torchvision.models import alexnet
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader
from dataloader import BasicDataset
from torch.autograd import Variable
import argparse
import numpy as np
import pickle

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'local': '/home/models/',
}

BATCH_SIZE = 4

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
    dataset = BasicDataset(args.labels_file, args.root_dir)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=BATCH_SIZE)
    net = get_feature_extractor()

    if torch.cuda.is_available():
        net = net.cuda()

    features_out = np.zeros((len(dataset), 4096))

    for i, samples in enumerate(dataloader):
        images = samples['image']
        if torch.cuda.is_available():
            images = images.cuda()
        images = Variable(images)
        features = net(images).cpu().data.numpy()
        features_out[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE] = features
        print(i)

    with open(args.out,'wb') as f:
        pickle.dump(features_out, f)
    return

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--labels_file', dest='labels_file')
    parser.add_argument('-r','--root_dir', dest='root_dir')
    parser.add_argument('-o','--out', dest='out')
    args = parser.parse_args()

    main(args)
