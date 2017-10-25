from sklearn.svm import LinearSVC
import progressbar
from sklearn.model_selection import GridSearchCV
import numpy as np
import pickle
import scipy.io as sio
import argparse
import os

def train(x, correct_i, wrong_i, n_images=None, c=None, loss="squared_hinge", w0=True):
    """
    Train SVM on a random samples of size n_images from x and y,
    using C as regularisation parameter.
    If w0=False, will train full model with all correct labels
    and randomly sampled incorrect labels, with parameters determined
    by 10 fold x-validation.
    Args:
        x (np.ndarray) : features of all samples
        correct_i (np.ndarray) : indexes of correct label in x
        wrong_i (np.ndarray) : indexes of incorrect label in x
        n_images (int) : number of images to sample for each category
                         not needed if w0=False
        c (float) : regularisation parameter to use for svm training
                    not needed if w0=False
        loss (str) : loss to train the SVM with, either "hinge" or
                    "squared hinge"
        w0 (bool) : whether training small or large svm
    Returns:
        model.coef_ (np.ndarray) : weight vector of the trained svm
        correct_i (np.ndarray) : training sample for correct label
        wrong_i (np.ndarray) : training sample for incorrect label
    """
    
    # set n_images to max if training w1
    n_images = n_images if w0 else len(correct_i)

    # sample correct images
    correct_i = np.random.choice(correct_i, size=n_images, replace=False)
    # sample wrong images
    wrong_i = np.random.choice(wrong_i, size=n_images, replace=False)

    x = np.concatenate((x[correct_i], x[wrong_i]), axis=0)

    y = np.asarray([1]*n_images + [0]*n_images)
    
    # shuffle them
    shuffle = np.random.permutation(2*n_images)

    if loss=="squared_hinge":
        dual=False
    else:
        dual=True
    
    if w0:
        model = LinearSVC(dual=dual, C=c, loss=loss)
        model.fit(x[shuffle], y[shuffle])
    else:
        svm = LinearSVC(dual=dual, loss=loss)

        # find best value of C by 10 x-validation
        gridsearch = GridSearchCV(svm, 
                param_grid={'C':[1e-2, 1e-1, 1, 10, 100]}, 
                cv=10,
                n_jobs=5)
        gridsearch.fit(x[shuffle], y[shuffle])

        model = gridsearch.best_estimator_
    
    return np.append(model.coef_, model.intercept_), correct_i, wrong_i

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-x')
    parser.add_argument('-y')
    parser.add_argument('-o')
    parser.add_argument('-s', '--stage', dest='s')
    parser.add_argument('--loss', type=str, default="squared_hinge")
    parser.add_argument('--train_split', type=float, default=0.9)

    args = parser.parse_args()

    with open(args.x,'rb') as f:
        x = pickle.load(f)

    # y = sio.loadmat(args.y)['labels'][0]
    with open(args.y, 'rb') as f:
        y = pickle.load(f) 

    labels = np.unique(y)
    

    progress = progressbar.ProgressBar(
            widgets=[progressbar.Bar(), ' ', progressbar.ETA()])

    # randomly pick labels for training and validation

    np.random.seed(42) # fix seed
    train_labels = np.random.choice(
        labels, 
        size=int(args.train_split*len(labels)),
        replace=False)
    np.random.seed() # re-seed for next random calls

    # training small samples
    if args.s == 'w0':

        # create folder if didnt exist
        if not os.path.exists(args.o):
            os.mkdir(args.o)
            os.mkdir(os.path.join(args.o, 'train'))
            os.mkdir(os.path.join(args.o, 'val'))
        
        
        for label in progress(labels):

            split = 'train' if label in train_labels else 'val'

            correct_labels = np.where(y==label)[0]

            # get labels of correct split that are not label
            split_i = np.where(np.isin(y, train_labels, invert=(split=='val')))[0]
            wrong_labels = np.intersect1d(np.where(y!=label)[0], split_i)

            counter = 0
            for n in range(1,21):
                for c in (1e-2, 1e-1, 1, 10, 100):
                    for s in range(5):
                        out_file = os.path.join(
                                                args.o, 
                                                split,
                                                'label_{}_{}'.format(
                                                    label, counter)
                                                )

                        if not os.path.exists(out_file):

                            w, correct_i, wrong_i = train(x, 
                                    correct_labels, wrong_labels, n, c,
                                    loss=args.loss)

                            sample = {'w0':w,
                                    'wrong_i':wrong_i,
                                    'correct_i':correct_i,
                                    'label':label
                                    }

                            with open(out_file, 'wb') as f:
                                pickle.dump(sample, f)
                        counter += 1
    
    # train large ground truth models
    elif args.s == 'w1':

        w1_matrix = []

        for label in progress(labels):

            split = 'train' if label in train_labels else 'val'

            correct_labels = np.where(y==label)[0]

            # get labels of correct split that are not label
            split_i = np.where(np.isin(y, train_labels, invert=(split=='val')))[0]
            wrong_labels = np.intersect1d(np.where(y!=label)[0], split_i)

            w, _ , _ = train(x,
                    correct_labels, wrong_labels, 
                    loss=args.loss, w0=False)
            w1_matrix.append(w)

        with open(os.path.join(
            args.o, 'w1.pickle'), 'wb') as f:

            pickle.dump(np.asarray(w1_matrix), f)
