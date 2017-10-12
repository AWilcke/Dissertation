import numpy as np
from sklearn import LinearSVC
from sklearn.model_selection import GridSearchCV
import torch
from torch.autograd import Variable

def retrain_svm(x, y, net):
    """
    Retrains a better SVM using the category-agnostic model regressor
    provided
    Args:
        - x (array-like) : (batch-size x feature size) array of features
        - y (array-like) : (batch-size) array of labels
        - net (pytorch model) : model regressor
    Returns:
        - svm (sklearn.LinearSVC) : the improved linear model
    """
    
    # find best parameters for bad svm
    shuffle = np.random.permutation(len(y))
    bad_svm = LinearSVC(dual=False)
    gridsearch = GridSearchCV(bad_svm,
            param_grid={'C':[1e-2, 1e-1, 1, 10, 100]},
            cv=10,
            n_jobs=5)
    gridsearch.fit(x[shuffle], y[shuffle])
    best_bad_svm = gridsearch.best_estimator_

    # create weight vector w0
    w0 = np.append(best_bad_svm.coef_, best_bad_svm.intercept_)
    w0 = Variable(torch.from_numpy(w0).float())
    if torch.cuda.is_available():
        w0 = w0.cuda()
    
    # run model regression
    t = net(w0).data.cpu().numpy()
    
    # retrain, biasing towards regressed model
    good_svm = LinearSVC(dual=False, regressed_w = t)
    good_gridsearch = GridSearchCV(good_svm,
            param_grid={'C':[1e-2, 1e-1, 1, 10, 100]},
            cv=10,
            n_jobs=5)
    good_gridsearch.fit(x[shuffle], y[shuffle])

    return good_gridsearch.best_estimator_
