# encoding: utf-8
"""
Read images and corresponding labels.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os


class CheXpertDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            csv_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(CheXpertDataset, self).__init__()
        file = pd.read_csv(csv_file)

        self.root_dir = root_dir
        self.images = file["ImageID"].values
        self.labels = file.iloc[:, 1:].values.astype(int)
        self.transform = transform

        print("Total # images:{}, labels:{}".format(len(self.images), len(self.labels)))

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        items = self.images[index]  # .split('/')
        # study = items[2] + '/' + items[3]
        image_name = os.path.join(self.root_dir, self.images[index])
        image = Image.open(image_name).convert("RGB")
        label = self.labels[index]
        # print(label)
        if self.transform is not None:
            image = self.transform(image)
        return items, index, image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.images)


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2
