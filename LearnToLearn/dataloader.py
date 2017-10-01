import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import scipy.io as sio
import pickle

class BasicDataset(Dataset):

    def __init__(self, labels_file, root_dir):
        """
        Args:
            labels_file (string) : the .mat file containing the labels
            root_dir (string) : the path where the dataset is stored
        """

        self.labels = sio.loadmat(os.path.join(root_dir, labels_file))['labels'][0]
        self.root_dir = root_dir
        self.preprocess = self._preprocess_fn()

    def _preprocess_fn(self):
        """
        Resize image to 256x256, take a center crop of 224x224,
        squeeze between 0 and 1 and normalise according to pretraining.
        Args:
            image (array) : image to preprocess
        Returns:
            image (array) : preprocessed image
        """
        data_transforms = transforms.Compose([
            transforms.Scale((256,256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
            ])

        return data_transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'jpg', 'image_{:05d}.jpg'.format(idx + 1))
        image = Image.open(img_name)
        image = self.preprocess(image)
        sample = {'image' : image, 'label' : self.labels[idx]}
        return sample

class SVMDataset(Dataset):

    def __init__(self, w0_folder, w1_file, features_file):
        """
        Args:
            w0_folder (string) : path to folder where w0 files are stored
            w1_file (string) : path to file where w1 files are stored
            features_file (string) : path to file where features are stored
        """

        self.w0_list = [os.path.join(w0_folder, filename) 
                for filename in os.listdir(w0_folder)]

        with open(w1_file, 'rb') as f:
            self.w1 = pickle.load(f)

        with open(features_file, 'rb') as f:
            self.features = pickle.load(f)

    def __len__(self):
        return len(self.w0_list)

    def __getitem__(self, idx):
        with open(self.w0_list[idx], 'rb') as f:
            sample = pickle.load(f)

        sample['w0'] = torch.from_numpy(sample['w0'])
        sample['w1'] = torch.from_numpy(self.w1[sample['label']-1])

        correct_i = torch.from_numpy(self.features[sample['correct_i']])
        wrong_i = torch.from_numpy(self.features[sample['wrong_i']])

        # take negative of wrong samples and concat with correct samples
        # makes the hinge loss easier, as it removes the need for the
        # correct label to determine the correct sign
        sample['train'] = torch.cat([correct_i, -wrong_i], 0)

        del sample['label'], sample['correct_i'], sample['wrong_i']

        return sample
