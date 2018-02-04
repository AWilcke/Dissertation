from torchvision import transforms as T
import torchvision
from torch.utils.data import Dataset
import pickle
import numpy as np
from pathlib import Path
from utils import dict_to_tensor_list
from torch.utils.data.dataloader import default_collate

class MNISTbyClass(Dataset):

    def __init__(self, root, index, label, n, train_labels=True, train_split=True):
        """
        Args:
            root (string) : path to folder where the "processed" folder is stored
            index (string) : path to "index.pickle" file
            label (int) : which label is the postive sample
            n (int) : number of images to sample
            train_labels (bool) : train or validation label set
            train_split (bool) : train or validation data split
        """

        relevant_labels = range(2,10) if train_labels else range(2)

        with open(index, 'rb') as f:
            index = pickle.load(f)[train_split]

        self.data = torchvision.datasets.MNIST(root, train=train_split, download=True,
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

        self.correct_i = np.random.choice(index[label], size=n, replace=False)
        self.wrong_i = np.random.choice(false, size=n, replace=False)

        # store set of indices and whether they are correct or not
        self.index = [(1, i) for i in self.correct_i] + [(0,i) for i in self.wrong_i]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        label, index = self.index[idx]
        img, _ = self.data[index]
        return (img, label)

class MLP_Dataset(Dataset):

    def __init__(self, w0, w1, mnist, train=True):

        split = 'train' if train else 'val'
        p = Path(w0) / split

        self.file_list = list(p.glob('*/*'))
        self.label_list = [x.parts[-2] for x in self.file_list]

        self.w1 = Path(w1) / split

        self.data = torchvision.datasets.MNIST(mnist, train=True, download=True,
                transform=T.Compose([
                    T.ToTensor(),
                    T.Lambda(lambda t : t.view(-1))
                    ])
                )
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # get info from w0
        with open(self.file_list[idx], 'rb') as f:
            sample = pickle.load(f)

        # transform state_dict into batcheable format

        sample['w0'] = dict_to_tensor_list(sample['weights'])

        # remove actual state_dict
        del sample['weights']

        # then associated w1 weights
        with open(self.w1 / self.label_list[idx] / "{}_0".format(self.label_list[idx]), 'rb') as f:
            sample['w1'] = pickle.load(f)

        sample['w1'] = dict_to_tensor_list(sample['w1'])

        sample['train'] = default_collate([(self.data[i][0], 1) for i in sample['correct_i']] + \
                [(self.data[i][0], 0) for i in sample['wrong_i']])

        sample['label'] = self.label_list[idx]

        del sample['wrong_i'], sample['correct_i']
        return sample
