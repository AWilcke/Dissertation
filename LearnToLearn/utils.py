import numpy as np
from sklearn import LinearSVC
import torch
from torch.autograd import Variable

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
