from torch import nn
from torch.autograd import Variable
import torch
from torch.autograd.gradcheck import zero_gradients
from models import SVMRegressor
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import pickle
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as KNN
from matplotlib import pyplot as plt
import numpy as np
import utils
from dataloader import FeatureDataset

plt.rcParams['image.cmap'] = 'plasma'
plt.switch_backend('agg')

parser = ArgumentParser()
parser.add_argument('--val_labels', nargs='+', type=int,
        help='Which labels to use as validation')
parser.add_argument('--features',
        help='path to features file')
parser.add_argument('--labels',
        help='path to labels file')

parser.add_argument('--model_path')
parser.add_argument('--regressor_path')
parser.add_argument('--adv_path')
parser.add_argument('--w1_path')
parser.add_argument('--label', type=int,
        help='label of current model')

parser.add_argument('--gradient', action='store_true')
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


def model_decision(net, args, regressed_net=None, w1_net=None, adv_net=None):

    step = 0.25

    dataset = FeatureDataset(args.features, args.labels, 
            args.label, 100, args.val_labels)

    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True,
            shuffle=False, num_workers=0)

    real = []
    boundary_og = []
    boundary_reg = []
    boundary_w1 = []
    boundary_adv = []

    acc_og = 0
    acc_reg = 0
    acc_w1 = 0
    acc_adv = 0
    total = 0
    for img, label in tqdm(dataloader):

        out, pred = follow_gradient(img, net, step)

        real.append((img.squeeze(0).numpy(), label[0]))
        boundary_og += out

        total += 1
        acc_og += (pred.long() == label)[0]

        if regressed_net is not None:
            out, pred_reg = follow_gradient(img, regressed_net, step)
            boundary_reg += out
            acc_reg += (pred_reg.long() == label)[0]

        if w1_net is not None:
            out, pred_w1 = follow_gradient(img, w1_net, step)
            boundary_w1 += out
            acc_w1 += (pred_w1.long() == label)[0]

        if adv_net is not None:
            out, pred_adv = follow_gradient(img, adv_net, step)
            boundary_adv += out
            acc_adv += (pred_adv.long() == label)[0]

    print(f'Original Accuracy: {acc_og/total:.3f}')
    print(f'Regressed Accuracy: {acc_reg/total:.3f}')
    print(f'Adversarial Accuracy: {acc_adv/total:.3f}')
    print(f'W1 Accuracy: {acc_w1/total:.3f}')

    return real, boundary_og, boundary_reg, boundary_w1, boundary_adv


def viz(net, args, regressed=None, w1_net=None, adv_net=None):

    real, boundary, boundary_reg, boundary_w1, boundary_adv = model_decision(
            net, args, regressed_net=regressed, w1_net=w1_net, adv_net=adv_net)

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

    if adv_net is not None:
        bound_adv_points = np.stack([x[0] for x in boundary_adv])
        bound_adv_labels = np.stack([x[1] for x in boundary_adv])
        bound_adv_points = pca.transform(bound_adv_points)

    # do original

    if args.nn:

        xmin, ymin = real_points.min(0)
        xmax, ymax = real_points.max(0)

        density = 300

        points = []
        for i in np.arange(xmin, xmax, abs(xmax-xmin)/density):
            for j in np.arange(ymin, ymax, abs(ymax-ymin)/density):
                points.append([i, j])

        points = np.stack(points)
    else:
        points = None

    plot_boundary(bound_points, bound_labels, real_points, real_labels, points, args, 'original')

    # optionally do regressed
    if regressed is not None:
        plot_boundary(bound_reg_points, bound_reg_labels, real_points, real_labels, points, args, 'regressed')

    # optionally do w1
    if w1_net is not None:
        plot_boundary(bound_w1_points, bound_w1_labels, real_points, real_labels, points, args, 'w1')

    # optionally do adv
    if adv_net is not None:
        plot_boundary(bound_adv_points, bound_adv_labels, real_points, real_labels, points, args, 'adv')


def plot_boundary(bound_points, bound_labels, real_points, real_labels, points, args, name):

    if args.nn:
        knn = KNN()

        print(f'Starting KNN - {name}')
        knn.fit(bound_points, (bound_labels > 0.5))
        background = knn.predict(points)
        plt.scatter(points[:, 0], points[:, 1], c=background, alpha=0.2)

    else:
        plt.scatter(bound_points[:, 0], bound_points[:, 1], c=bound_labels, s=300, alpha=0.2)

    plt.scatter(real_points[:, 0], real_points[:, 1], c=real_labels, linewidths=1, edgecolors='black')
    plt.axis('off')
    plt.savefig(f'{args.save}_{name}.png', bbox_inches = 'tight', pad_inches = 0)


if __name__ == '__main__':

    args = parser.parse_args()

    #### LOAD NET ####
    net = nn.Sequential(
            nn.Linear(4096,1),
            nn.Sigmoid(),
            )
    # print(net)

    with open(args.model_path, 'rb') as f:
        d = pickle.load(f)
        w0 = torch.from_numpy(d['w0']).float()
        num_images = len(d['wrong_i'])

        net = utils.tensor_to_net(w0, net)

    print(f'Trained with {num_images} images')
    net = net.cuda()

    #### REGRESS NET ####
    regressed_net = None
    # init regressor
    regressor = SVMRegressor()
    regressor.eval()
    # print(regressor)
    regressor.load_state_dict(
            torch.load(args.regressor_path))

    # this will store regressed version
    regressed_net = nn.Sequential(
            nn.Linear(4096,1),
            nn.Sigmoid(),
            )
    
    # run through regression
    w0 = Variable(w0.unsqueeze_(0))
    if torch.cuda.is_available():
        regressed_net = regressed_net.cuda()
        regressor = regressor.cuda()
        w0 = w0.cuda()

    w1 = regressor(w0).data
    w1.squeeze_(0)

    # copy to new identical net
    regressed_net = utils.tensor_to_net(w1, regressed_net)

    #### REGRESS ADV NET ####
    regressor.load_state_dict(
            torch.load(args.adv_path))

    # this will store regressed version
    regressed_adv = nn.Sequential(
            nn.Linear(4096,1),
            nn.Sigmoid(),
            )
    
    # run through regression
    if torch.cuda.is_available():
        regressed_adv = regressed_adv.cuda()

    w1 = regressor(w0).data
    w1.squeeze_(0)

    # copy to new identical net
    regressed_adv = utils.tensor_to_net(w1, regressed_adv)

    #### W1 NET ####
    w1_net = None
    if args.w1_path:
        w1_net = nn.Sequential(
                nn.Linear(4096,1),
                nn.Sigmoid(),
                )
        with open(args.w1_path, 'rb') as f:
            w1 = torch.from_numpy(pickle.load(f)[args.label]).float()
        w1_net = utils.tensor_to_net(w1, w1_net)

        if torch.cuda.is_available():
            w1_net = w1_net.cuda()

    viz(net, args, regressed_net, w1_net, regressed_adv)
