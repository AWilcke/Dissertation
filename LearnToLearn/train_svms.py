from sklearn.svm import LinearSVC
import progressbar
from sklearn.model_selection import GridSearchCV
import numpy as np
import pickle
import scipy.io as sio
import argparse
import os

def train(x, correct_i, wrong_i, n_images=None, c=None, w0=True):
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
    
    if w0:
        model = LinearSVC(loss='hinge', C=c)
        model.fit(x[shuffle], y[shuffle])
    else:
        svm = LinearSVC(loss='hinge')

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

    args = parser.parse_args()

    with open(args.x,'rb') as f:
        x = pickle.load(f)

    y = sio.loadmat(args.y)['labels'][0]
    labels = np.unique(y)

    progress = progressbar.ProgressBar(
            widgets=[progressbar.Bar(), ' ', progressbar.ETA()])

    # training small samples
    if args.s == 'w0':

        # create folder if didnt exist
        if not os.path.exists(args.o):
            os.mkdir(args.o)
        
        # if did exist, and not empty, 
        # find which label to start from
        elif os.listdir(args.o):
            last_file = sorted(os.listdir(args.o), key=lambda x : int(x.split('_')[1]))[-1]
            last_label = int(last_file.split('_')[1])
            labels = [label for label in labels if label>=last_label]
            print("Resuming from label {}".format(last_label))
        
        for label in progress(labels):

            correct_labels = np.where(y==label)[0]
            wrong_labels = np.where(y!=label)[0]
            # y_test = np.zeros_like(y)
            # y_test[correct_labels] = 1
            # y_test[wrong_labels] = 0


            for n in range(1,21):
                for c in (1e-2, 1e-1, 1, 10, 100):
                    for s in range(5):
                        out_file = os.path.join(
                                                args.o, 
                                                'label_{}_n_{}_c_{:.0e}_{}.pickle'.format(
                                                    label, n, c, s)
                                                )

                        if not os.path.exists(out_file):

                            w, correct_i, wrong_i = train(x, 
                                    correct_labels, wrong_labels, n, c)
                            # sample = np.random.choice(len(y), size=1000)
                            # print(w.score(x[sample], y_test[sample]))

                            sample = {'w0':w,
                                    'wrong_i':wrong_i,
                                    'correct_i':correct_i,
                                    'label':label
                                    }

                            with open(out_file, 'wb') as f:
                                pickle.dump(sample, f)
    
    # train large ground truth models
    elif args.s == 'w1':

        w1_matrix = []

        for label in progress(labels):

            correct_labels = np.where(y==label)[0]
            wrong_labels = np.where(y!=label)[0]
            w, _ , _ = train(x,
                    correct_labels, wrong_labels, w0=False)
            w1_matrix.append(w)

        with open(os.path.join(
            args.o, 'w1.pickle'), 'wb') as f:

            pickle.dump(np.asarray(w1_matrix), f)
