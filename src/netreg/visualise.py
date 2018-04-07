from torch import nn
from torch.autograd import Variable
import torch
from torch.autograd.gradcheck import zero_gradients
from dataset import MNISTbyClass
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from models import MLP_100
import pickle
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as KNN
from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['image.cmap'] = 'plasma'

parser = ArgumentParser()
parser.add_argument('--mnist')
parser.add_argument('--extended', action='store_true')
parser.add_argument('--val_labels', nargs='+', type=int,
        help='Which labels to use as validation')
parser.add_argument('--index')
parser.add_argument('--label', type=int,
        help='label of current model')
parser.add_argument('--model_path')
parser.add_argument('--gradient', action='store_true')
parser.add_argument('--boundary', action='store_true')
parser.add_argument('--nn', action='store_true')
parser.add_argument('--save', type=str)


def follow_gradient(img, net, alpha):
    boundary = []

    loss = nn.BCELoss()

    y = Variable(torch.FloatTensor(1, 1))

    if torch.cuda.is_available():
        img = img.cuda()
        y = y.cuda()

    x = Variable(img, requires_grad=True)
    zero_gradients(x)

    # get initial prediction
    out = net(x)
    pred = (out.data[0] > 0.5).cpu()

    boundary.append((img.cpu().squeeze(0).numpy(), out.data[0, 0]))

    # copy prediction into y
    y.data.copy_(pred)

    for _ in range(10):
        error = loss(out, y)
        error.backward()

        gradient = torch.sign(x.grad.data)

        gradient_mask = alpha * gradient

        # create updated img
        output = x.data + gradient_mask
        output.clamp_(0., 1.)

        # put step in x
        x.data.copy_(output)

        # repeat the process
        zero_gradients(x)
        out = net(x)
        boundary.append((output.cpu().squeeze(0).numpy(), out.data[0, 0]))

    return boundary, pred


def gradient_boundary(img, net, alpha):
    loss = nn.BCELoss()

    y = Variable(torch.FloatTensor(1, 1))

    if torch.cuda.is_available():
        img = img.cuda()
        y = y.cuda()

    x = Variable(img, requires_grad=True)
    zero_gradients(x)

    # get initial prediction
    out = net(x)
    pred = (out.data[0] > 0.5).cpu()

    # copy prediction into y
    y.data.copy_(pred)

    for _ in range(100):
        error = loss(out, y)
        error.backward()

        gradient = torch.sign(x.grad.data)

        gradient_mask = alpha * gradient

        # create updated img
        output = x.data + gradient_mask
        output.clamp_(0., 1.)

        # put step in x
        x.data.copy_(output)

        # repeat the process
        zero_gradients(x)
        out = net(x)

        # return if on decision boundary
        if (out.data - 0.5).abs()[0, 0] < 0.01:
            return [(output.cpu().squeeze(0).numpy(), 0.5)], pred

    return [], pred


def random_jitter(img, net, sigma):

    boundary = []

    noise = Variable(torch.zeros_like(img))

    if torch.cuda.is_available():
        img = img.cuda()
        noise = noise.cuda()

    x = Variable(img)
    for _ in range(10):
        noise.data.normal_(0, sigma)
        jittered = x + noise

        out = net(jittered)
        boundary.append((jittered.data.cpu().squeeze(0).numpy(), out.data[0, 0]))

    return boundary


def model_decision(net, args):

    dataset = MNISTbyClass(args.mnist, args.index, 
            args.label, 400,
            relevant_labels=args.val_labels, train_split=False,
            extended=args.extended)

    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True,
            shuffle=False, num_workers=0)

    real = []
    boundary = []

    acc = 0
    total = 0
    for img, label in tqdm(dataloader):
        if args.gradient:
            out, pred = follow_gradient(img, net, 0.1)
        elif args.boundary:
            out, pred = gradient_boundary(img, net, 1e-2)
        else:
            out = random_jitter(img, net, 0.1)

        real.append((img.squeeze(0).numpy(), label[0]))
        boundary += out

        total += 1
        acc += (pred.long() == label)[0]

    print(f'Accuracy: {acc/total:.3f}')

    return real, boundary


def viz(args):
    net = MLP_100()

    with open(args.model_path, 'rb') as f:
        d = pickle.load(f)['weights']
        net.load_state_dict(d)

    if torch.cuda.is_available():
        net = net.cuda()

    real, boundary = model_decision(net, args)

    real_points = np.stack([x[0] for x in real])
    real_labels = np.stack([x[1] for x in real])

    bound_points = np.stack([x[0] for x in boundary])
    bound_labels = np.stack([x[1] for x in boundary])

    pca = PCA(n_components=2)

    pca.fit(real_points)

    real_points = pca.transform(real_points)
    bound_points = pca.transform(bound_points)

    if args.nn:

        xmin, ymin = real_points.min(0)
        xmax, ymax = real_points.max(0)

        points = []
        for i in np.arange(xmin, xmax, 0.05):
            for j in np.arange(ymin, ymax, 0.05):
                points.append([i, j])

        points = np.stack(points)
        knn = KNN()

        print('Starting KNN')
        knn.fit(bound_points, (bound_labels > 0.5))
        background = knn.predict(points)
        plt.scatter(points[:, 0], points[:, 1], c=background, alpha=0.2)

    else:
        plt.scatter(bound_points[:, 0], bound_points[:, 1], c=bound_labels, s=300, alpha=0.2)

    plt.scatter(real_points[:, 0], real_points[:, 1], c=real_labels, linewidths=1, edgecolors='black')

    if args.save:
        plt.savefig(f'{args.save}.png')
    else:
        plt.show()


if __name__ == '__main__':

    args = parser.parse_args()
    viz(args)
