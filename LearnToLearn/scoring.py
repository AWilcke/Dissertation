import numpy as np
from sklearn.svm import LinearSVC
import torch
from torch.autograd import Variable
import argparse
from collections import defaultdict
import pickle
from models import SVMRegressor
import os
import progressbar
from adv import dropout_train 

def retrain_svm(x, y, net, c=1):
    """
    Retrains a better SVM using the category-agnostic model regressor
    provided
    Args:
        - x (array-like) : (batch-size x feature size) array of features
        - y (array-like) : (batch-size) array of labels
        - net (pytorch model) : model regressor
        -c (float) : regularisation parameter for the svm
    Returns:
        - svm (sklearn.LinearSVC) : the improved linear model
    """
    
    shuffle = np.random.permutation(len(y))
    bad_svm = LinearSVC(dual=False, C=c)
    bad_svm.fit(x[shuffle], y[shuffle])

    # create weight vector w0
    w0 = np.append(bad_svm.coef_, bad_svm.intercept_)
    w0 = Variable(torch.from_numpy(w0).float())
    if torch.cuda.is_available():
        w0 = w0.cuda()
    
    # run model regression
    t = net(w0).data.cpu().numpy()
    
    # retrain, biasing towards regressed model
    good_svm = LinearSVC(dual=False, regressed_w = t, C=c)
    good_svm.fit(x[shuffle], y[shuffle])

    return good_svm

def score_svm(sample, data, labels, net, use_w0=False, refit=False, num_samples=10, svm_params=None):
    """
    Scores an SVM using left-out data.
    Args:
        - sample (dict) : sample dict from train_svms.py
        - data (array-like) : (n_samples x n_features) matrix containing features
        - labels (array-like) : (1 x n_samples) array containing labels of samples
        - use_w0 (bool) : whether to run the scoring with the given w0, as is,
                          or to run the model regression
        - refit (bool) : whether to run the refinement step of the process
        - num_samples (int) : how many times to sample negative testing data
        - svm_params (dict) : dict containing initialisation parameters for the svm
    Returns:
        - len(correct_i) (int) : how many positive/negative images the svm was trained on
        - acc : average testing accuracy across all samples
    """

    w0 = sample['w0']
    correct_i = sample['correct_i']
    wrong_i = sample['wrong_i']
    label = sample['label']

    if svm_params is None:
        svm_params = {}

    if not use_w0:
        if net is None:
            raise Exception("Must pass a net if use_w0=False")

        w0 = Variable(torch.from_numpy(w0).float())
        w0 = w0.view(1, -1)
        if torch.cuda.is_available():
            w0 = w0.cuda()
        w0 = net(w0)
        w0 = w0.data.cpu().numpy().reshape(-1)


    if refit:
        svm = LinearSVC(regressed_w=w0, **svm_params)
        shuffle = np.random.permutation(2*len(correct_i))
        y = np.array([1] * len(correct_i) + [0] * len(wrong_i))
        index = np.concatenate((correct_i, wrong_i))
        svm.fit(data[index][shuffle], y[shuffle])
    else:
        svm = LinearSVC(**svm_params)
        svm.coef_ = w0[:-1].reshape(1,-1)
        svm.intercept_ = w0[-1]
        svm.classes_ = np.array([0,1])
    
    # get all samples that were not used for training this SVM
    test_correct_i = list(filter(lambda x : x not in correct_i, np.where(labels==label)[0]))
    test_wrong_i = list(filter(lambda x : x not in wrong_i, np.where(labels!=label)[0]))
    
    acc = []
    for i in range(num_samples):
        test_i = np.concatenate((test_correct_i, 
                np.random.choice(test_wrong_i, size=len(test_correct_i), replace=False)))

        score = svm.score(data[test_i], [1]*len(test_correct_i) + [0]*len(test_correct_i)) 
        acc.append(score)
    
    return len(correct_i), np.mean(acc)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-x')
    parser.add_argument('-y')
    parser.add_argument('-o')
    parser.add_argument('--ckpt')
    parser.add_argument('--val_path')
    parser.add_argument('--usew0', action='store_true')
    parser.add_argument('--refit', action='store_true')
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('-C', type=float)
    parser.add_argument('--loss', type=str, default="squared_hinge")
    parser.add_argument('--dropout', type=float, default=0)

    args = parser.parse_args()


    with open(args.x, 'rb') as f:
        x = pickle.load(f)

    # y = loadmat(args.y)['labels'][0]
    with open(args.y,'rb') as f:
        y = pickle.load(f)

    if args.usew0:
        net = None
    else:
        net = SVMRegressor(dropout=args.dropout)
        net.load_state_dict(torch.load(args.ckpt))

        if torch.cuda.is_available():
            net = net.cuda()
        net.eval()
        net.apply(dropout_train)

    params = {'loss' : args.loss, 'dual':False}
    if args.C:
        params['C'] = args.C

    scores = defaultdict(list)
    p = progressbar.ProgressBar(widgets=[progressbar.ETA(), ' ', progressbar.Percentage(), ' ', progressbar.Bar()])
    for sample in p([os.path.join(args.val_path, w) for w in os.listdir(args.val_path)]):
        with open(sample, 'rb') as f:
            s = pickle.load(f)

        n, acc = score_svm(s, x, y, net, args.usew0, args.refit, args.n_samples, params)

        scores[n].append(acc)

    with open(args.o, 'wb') as f:
        pickle.dump(scores, f)
