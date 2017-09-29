from sklearn.svm import LinearSVC
import numpy as np
import pickle
import scipy.io as sio
import argparse
import os

def train(x, correct_i, wrong_i, n_images, c):
    """
    Train SVM on a random samples of size n_images from x and y,
    using C as regularisation parameter.
    Args:
        x (np.ndarray) : features of all samples
        correct_i (np.ndarray) : indexes of correct label in x
        wrong_i (np.ndarray) : indexes of incorrect label in x
        n_images (int) : number of images to sample for each category
        c (float) : regularisation parameter to use for svm training
    Returns:
        model.coef_ (np.ndarray) : weight vector of the trained svm
        correct_i (np.ndarray) : training sample for correct label
        wrong_i (np.ndarray) : training sample for incorrect label
    """

    # sample correct images
    correct_i = np.random.choice(correct_i, size=n_images, replace=False)
    # sample wrong images
    wrong_i = np.random.choice(wrong_i, size=n_images, replace=False)

    x = np.concatenate((x[correct_i], x[wrong_i]), axis=0)

    y = np.asarray([1]*n_images + [0]*n_images)
    
    # shuffle them
    shuffle = np.random.permutation(2*n_images)
    
    model = LinearSVC(C=c)
    model.fit(x[shuffle], y[shuffle])
    
    return np.append(model.coef_, model.intercept_), correct_i, wrong_i

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

        for n in range(1,20):
            for c in (1e-2, 1e-1, 1, 10, 100):
                for s in range(5):
                    w, correct_i, wrong_i = train(x, correct_labels, wrong_labels, n, c)
                    # sample = np.random.choice(len(y), size=1000)
                    # print(w.score(x[sample], y_test[sample]))

                    sample = {'w0':w,
                            'wrong_i':wrong_i,
                            'correct_i':correct_i,
                            'label':label
                            }

                    with open(
                            os.path.join(
                                args.o, 
                                'label_{}_n_{}_c_{}_{}'.format(label, n, c, s)
                                ),
                            'wb') as f:
                        pickle.dump(sample, f)
