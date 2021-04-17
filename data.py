import cv2
import numpy as np
import os
import torch
import random
from torch.utils.data import Dataset
from pascal_voc_loader import PascalVOCLoader
from torch.utils.data.sampler import SubsetRandomSampler


class RawDataset:
    def __init__(self,
                 root_dir,
                 augmentations=None,
                 ds_split=0.8,
                 num_workers=1,
                 output_dims=224,
                 output_channels=3,
                 shuffle=True,
                 batch_size_dict=None):
        self.name = os.path.basename(os.path.normpath(root_dir))
        self.output_dims = output_dims
        self.output_channels = output_channels
        self._ds_split = ds_split
        self.root_dir = root_dir
        self.num_workers = num_workers
        if not batch_size_dict:
            self.batch_size = {'train': 1, 'test': 1}
        self.batch_size = batch_size_dict

        self.dataset = PascalVOCLoader(
            root_dir,
            augmentations,
            output_dim=224,
            mode='classification')

        # Creating data indices for training and validation splits:
        dataset_size = len(self.dataset)
        validation_split = 1 - ds_split
        random_seed = 42
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size['train'],
            num_workers=self.num_workers,
            sampler=train_sampler)
        validation_loader = torch.utils.data.DataLoader(
            self.dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size['test'],
            sampler=valid_sampler)

        self.datasets = {'train': train_loader, 'test': validation_loader}
