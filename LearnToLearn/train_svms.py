from sklearn.svm import LinearSVC
import numpy as np
import pickle
import scipy.io as sio
import argparse
from itertools import chain
import os

def train(x_correct, x_wrong, n_images, c):
    """
    Train SVM on a random samples of size n_images from x and y,
    using C as regularisation parameter.
    """

    # sample correct images
    xs_c = x_correct[np.random.choice(len(x_correct), size=n_images, replace=False)]
    # sample wrong images
    xs_w = x_wrong[np.random.choice(len(x_wrong), size=n_images, replace=False)]

    x = np.concatenate((xs_c, xs_w), axis=0)

    y = np.asarray([1]*n_images + [0]*n_images)
    
    # shuffle them
    shuffle = np.random.permutation(2*n_images)
    
    model = LinearSVC(C=c)
    model.fit(x[shuffle], y[shuffle])

    return model.coef_

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-x')
    parser.add_argument('-y')
    parser.add_argument('-o')

    args = parser.parse_args()

    with open(args.x,'rb') as f:
        x = pickle.load(f)

    y = sio.loadmat(args.y)['labels'][0]
    labels = np.unique(y)

    for label in labels:
        correct_labels = np.where(y==label)[0]
        wrong_labels = np.where(y!=label)[0]
        # y_test = np.zeros_like(y)
        # y_test[correct_labels] = 1
        # y_test[wrong_labels] = 0

        for n in chain(range(1,10), range(10,42, 2)):
            for c in (1e-2, 1e-1, 1, 10, 100):
                for s in range(5):
                    w = train(x[correct_labels], x[wrong_labels], n, c)
                    # sample = np.random.choice(len(y), size=1000)
                    # print(w.score(x[sample], y_test[sample]))

                    with open(os.path.join(args.o, 'label_{}_n_{}_c_{}_{}'.format(label, n, c, s)), 'wb') as f:
                        f.write(w.dumps())
