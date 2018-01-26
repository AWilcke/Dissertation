from torchvision import transforms as T
import torchvision
from torch.utils.data import Dataset
import pickle
import numpy as np

class MNISTbyClass(Dataset):

    def __init__(self, root, index, label, n, train_labels=True, train_split=True):
        """
        Args:
            root (string) : path to folder where the "processed" folder is stored
            index (string) : path to "index.pickle" file
            label (int) : which label is the postive sample
            n (int) : number of images to sample
            train (bool) : train or validation set
        """

        relevant_labels = range(2,10) if train_labels else range(2)

        with open(index, 'rb') as f:
            index = pickle.load(f)[train_split]

        self.data = torchvision.datasets.MNIST(root, train=train_split,
                transform=T.Compose([
                    T.ToTensor(),
                    T.Lambda(lambda t : t.view(-1))
                    ])
                )

        # get indices of other nums
        false = []
        for num, ind in index.items():
            if num != label and num in relevant_labels:
                false += ind

        correct_i = np.random.choice(index[label], size=n, replace=False)
        wrong_i = np.random.choice(false, size=n, replace=False)

        # store set of indices and whether they are correct or not
        self.index = [(1, i) for i in correct_i] + [(0,i) for i in wrong_i]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        label, index = self.index[idx]
        img, _ = self.data[index]
        return (img, label)
