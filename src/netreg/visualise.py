from torch import nn
from torch.autograd import Variable
import torch
from torch.autograd.gradcheck import zero_gradients
from dataset import MNISTbyClass
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from models import MLP_100, ConvNet, \
        ConvNetRegressor, ConvConvNetRegressor, \
        VAEConvRegressor, MLP_Regressor
import pickle
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as KNN
from matplotlib import pyplot as plt
import numpy as np
import utils

plt.rcParams['image.cmap'] = 'plasma'
plt.switch_backend('agg')

models = {
        'mlp': MLP_100,
        'conv': ConvNet,
        }

regressors = {
        'convreg': ConvNetRegressor,
        'convconv': ConvConvNetRegressor,
        'vae': VAEConvRegressor,
        'mlp': MLP_Regressor,
        }

parser = ArgumentParser()
parser.add_argument('--model', choices=models.keys())
parser.add_argument('--regressor', choices=regressors.keys())
parser.add_argument('--mnist')
parser.add_argument('--extended', action='store_true')
parser.add_argument('--val_labels', nargs='+', type=int,
        help='Which labels to use as validation')
parser.add_argument('--index')
parser.add_argument('--label', type=int,
        help='label of current model')

parser.add_argument('--no_bias', action='store_true')
parser.add_argument('--h1', type=float, default=0.75)
parser.add_argument('--h2', type=float, default=0.5)

parser.add_argument('--model_path')
parser.add_argument('--regressor_path')
parser.add_argument('--w1_path')

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


def model_decision(net, args, regressed_net=None, w1_net=None):

    dataset = MNISTbyClass(args.mnist, args.index, 
            args.label, 400,
            relevant_labels=args.val_labels, train_split=False,
            extended=args.extended)

    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True,
            shuffle=False, num_workers=0)

    real = []
    boundary_og = []
    boundary_reg = []
    boundary_w1 = []

    acc_og = 0
    acc_reg = 0
    acc_w1 = 0
    total = 0
    for img, label in tqdm(dataloader):

        out, pred = follow_gradient(img, net, 0.1)

        real.append((img.squeeze(0).numpy(), label[0]))
        boundary_og += out

        total += 1
        acc_og += (pred.long() == label)[0]

        if regressed_net is not None:
            out, pred_reg = follow_gradient(img, regressed_net, 0.1)
            boundary_reg += out
            acc_reg += (pred_reg.long() == label)[0]

        if w1_net is not None:
            out, pred_w1 = follow_gradient(img, w1_net, 0.1)
            boundary_w1 += out
            acc_w1 += (pred_w1.long() == label)[0]

    print(f'Original Accuracy: {acc_og/total:.3f}')
    print(f'Regressed Accuracy: {acc_reg/total:.3f}')
    print(f'W1 Accuracy: {acc_w1/total:.3f}')

    return real, boundary_og, boundary_reg, boundary_w1


def viz(net, args, regressed=None, w1_net=None):

    real, boundary, boundary_reg, boundary_w1 = model_decision(
            net, args, regressed_net=regressed, w1_net=w1_net)

    real_points = np.stack([x[0] for x in real])
    real_labels = np.stack([x[1] for x in real])

    bound_points = np.stack([x[0] for x in boundary])
    bound_labels = np.stack([x[1] for x in boundary])

    pca = PCA(n_components=2)

    pca.fit(real_points)

    real_points = pca.transform(real_points)
    bound_points = pca.transform(bound_points)

    if regressed is not None:
        bound_reg_points = np.stack([x[0] for x in boundary_reg])
        bound_reg_labels = np.stack([x[1] for x in boundary_reg])
        bound_reg_points = pca.transform(bound_reg_points)

    if w1_net is not None:
        bound_w1_points = np.stack([x[0] for x in boundary_w1])
        bound_w1_labels = np.stack([x[1] for x in boundary_w1])
        bound_w1_points = pca.transform(bound_w1_points)

    # do original

    if args.nn:

        xmin, ymin = real_points.min(0)
        xmax, ymax = real_points.max(0)

        points = []
        for i in np.arange(xmin, xmax, 0.05):
            for j in np.arange(ymin, ymax, 0.05):
                points.append([i, j])

        points = np.stack(points)
        knn = KNN()

        print('Starting KNN - original')
        knn.fit(bound_points, (bound_labels > 0.5))
        background = knn.predict(points)
        plt.scatter(points[:, 0], points[:, 1], c=background, alpha=0.2)

    else:
        plt.scatter(bound_points[:, 0], bound_points[:, 1], c=bound_labels, s=300, alpha=0.2)

    plt.scatter(real_points[:, 0], real_points[:, 1], c=real_labels, linewidths=1, edgecolors='black')

    plt.savefig(f'{args.save}_original.png')

    # optionally do regressed
    if regressed is not None:

        if args.nn:

            knn = KNN()

            print('Starting KNN - regressed')
            knn.fit(bound_reg_points, (bound_reg_labels > 0.5))
            background = knn.predict(points)
            plt.scatter(points[:, 0], points[:, 1], c=background, alpha=0.2)

        else:
            plt.scatter(bound_reg_points[:, 0], bound_reg_points[:, 1], c=bound_labels, s=300, alpha=0.2)

        plt.scatter(real_points[:, 0], real_points[:, 1], c=real_labels, linewidths=1, edgecolors='black')

        plt.savefig(f'{args.save}_regressed.png')

    # optionally do w1
    if w1_net is not None:

        if args.nn:

            knn = KNN()

            print('Starting KNN - w1')
            knn.fit(bound_w1_points, (bound_w1_labels > 0.5))
            background = knn.predict(points)
            plt.scatter(points[:, 0], points[:, 1], c=background, alpha=0.2)

        else:
            plt.scatter(bound_w1_points[:, 0], bound_w1_points[:, 1], c=bound_labels, s=300, alpha=0.2)

        plt.scatter(real_points[:, 0], real_points[:, 1], c=real_labels, linewidths=1, edgecolors='black')

        plt.savefig(f'{args.save}_w1.png')


if __name__ == '__main__':

    args = parser.parse_args()

    #### LOAD NET ####
    net = models[args.model](bias=not args.no_bias)
    # print(net)

    with open(args.model_path, 'rb') as f:
        d = pickle.load(f)
        net.load_state_dict(d['weights'])
        num_images = len(d['wrong_i'])

    print(f'Trained with {num_images} images')
    net = net.cuda()

    #### REGRESS NET ####
    regressed_net = None
    if args.regressor:
        # init regressor
        regressor = regressors[args.regressor](
                bn=False, 
                regressor=True,
                h1=args.h1,
                h2=args.h2
                )
        # print(regressor)
        regressor.load_state_dict(
                torch.load(args.regressor_path))

        # this will store regressed version
        regressed_net = models[args.model](bias=not args.no_bias)
        
        # run through regression
        w0 = [Variable(x.unsqueeze(0)) for x in utils.dict_to_tensor_list(net.state_dict())]
        if torch.cuda.is_available():
            regressed_net = regressed_net.cuda()
            regressor = regressor.cuda()
            w0 = [x.cuda() for x in w0]
        w1 = regressor(w0)
        w1 = [x.data.squeeze(0) for x in w1]

        # copy to new identical net
        utils.copy_tensor_list_to_net(w1, regressed_net)

    #### W1 NET ####
    w1_net = None
    if args.w1_path:
        w1_net = models[args.model](bias=not args.no_bias)
        with open(args.w1_path, 'rb') as f:
            d = pickle.load(f)
            w1_net.load_state_dict(d['weights'])
        w1_net = w1_net.cuda()

    viz(net, args, regressed_net, w1_net)
