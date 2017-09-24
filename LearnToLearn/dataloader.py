import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import scipy.io as sio

#TODO resize + centercrop

class BasicDataset(Dataset):

    def __init__(self, labels_file, root_dir):
        """
        Args:
            labels_file (string) : the .mat file containing the labels
            root_dir (string) : the path where the dataset is stored
        """

        self.labels = sio.loadmat(os.path.join(root_dir, labels_file))['labels'][0]
        self.root_dir = root_dir

    def preprocess(self):
        """
        Resize image to 256x256, and take a center crop of 224x224.
        Args:
            image (array) : image to preprocess
        Returns:
            image (array) : preprocessed image
        """
        data_transforms = transforms.Compose([
            transforms.Scale((256,256)),
            transforms.CenterCrop(224),
            transforms.ToTensor()
            ])

        return data_transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'jpg', 'image_{:05d}.jpg'.format(idx + 1))
        image = Image.open(img_name)
        image = self.preprocess()(image)
        sample = {'image' : image, 'label' : self.labels[idx]}
        return sample
