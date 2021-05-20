import random

import PIL.Image
import torch
import os
import numpy as np
from torch.utils import data
from torch.utils.data import SequentialSampler, RandomSampler
from sys import maxsize as maxint


def build_balanced_dataloader(dataset, labels, target_weight=None, batch_size=1, num_workers=0, steps_per_epoch=500):
    assert len(dataset) == len(labels)
    labels = np.asarray(labels)
    ulabels, label_count = np.unique(labels, return_counts=True)
    assert (ulabels == list(range(len(ulabels)))).all()
    balancing_weight = 1 / label_count
    target_weight = target_weight if target_weight is not None else np.ones(len(ulabels))
    assert len(target_weight) == len(ulabels)

    from torch.utils.data import WeightedRandomSampler
    #num_samples = steps_per_epoch * batch_size
    weighted_sampler = WeightedRandomSampler(
        weights=(target_weight * balancing_weight)[labels],
        num_samples=len(labels),
        replacement=True
    )
    loader = torch.utils.data.DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        sampler=weighted_sampler,
                        collate_fn=my_collate)
    return loader


def load_func(path, file, all_files):
    label = 0 if 'Neg' in path else 1
    path_to_file = os.path.join(path, file)
    p_image = PIL.Image.open(path_to_file)
    np_image = np.asarray(p_image)
    tensor_image = torch.tensor(np_image)
    img_name, format = str(file).split('.')
    mask_file = img_name+'m'+'.'+format
    if all_files is not None and label == 1 and mask_file in all_files:
        path_to_mask = os.path.join(path, mask_file)
        p_mask = PIL.Image.open(path_to_mask)
        np_mask = np.asarray(p_mask)
        tensor_mask = torch.tensor(np_mask)
        return tensor_image, tensor_mask, label
    return tensor_image, -1, label


class MedT_Train_Data(data.Dataset):
    def __init__(self, root_dir='train', loader=load_func):
        self.pos_root_dir = root_dir+'Pos/'
        self.neg_root_dir = root_dir + 'Neg/'
        all_neg_files = os.listdir(self.neg_root_dir)
        all_pos_files = os.listdir(self.pos_root_dir)
        pos_cl_images = [file for file in all_pos_files if 'm' not in file]
        self.all_files = all_pos_files+all_neg_files
        self.all_cl_images = pos_cl_images+all_neg_files
        self.pos_num_of_samples = len(pos_cl_images)
        self.loader = loader

    def __len__(self):
        return len(self.all_cl_images)

    def __getitem__(self, index):
        if index < self.pos_num_of_samples:
            return self.loader(self.pos_root_dir, self.all_cl_images[index], self.all_files)
        return self.loader(self.neg_root_dir, self.all_cl_images[index], None)

    def positive_len(self):
        return self.pos_num_of_samples


class MedT_Test_Data(data.Dataset):
    def __init__(self, root_dir='train', loader=load_func):
        self.pos_root_dir = root_dir+'Pos/'
        self.neg_root_dir = root_dir + 'Neg/'
        self.all_files = os.listdir(self.pos_root_dir)+os.listdir(self.neg_root_dir)
        self.pos_num_of_samples = len(os.listdir(self.pos_root_dir))
        self.loader = loader

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        if index < self.pos_num_of_samples:
            return self.loader(self.pos_root_dir, self.all_files[index], None)
        return self.loader(self.neg_root_dir, self.all_files[index], None)

    def positive_len(self):
        return self.pos_num_of_samples


def my_collate(batch):
    imgs, masks, labels = zip(*batch)
    return imgs, masks, labels

class MedT_Loader():
    def __init__(self, root_dir, target_weight, batch_size=1):
        self.train_dataset = MedT_Train_Data(root_dir+'training/')
        self.test_dataset = MedT_Test_Data(root_dir + 'validation/')

        #train_sampler = RandomSampler(self.train_dataset, num_samples=maxint,
        #                              replacement=True)
        test_sampler = SequentialSampler(self.test_dataset)

        '''
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=1,
            num_workers=0,
            sampler=train_sampler)
        '''
        ones = torch.ones(self.train_dataset.positive_len())
        labels = torch.zeros(len(self.train_dataset))
        labels[0:len(ones)] = ones
        train_loader = build_balanced_dataloader(self.train_dataset, labels.int(), target_weight, batch_size)

        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            num_workers=0,
            batch_size=batch_size,
            sampler=test_sampler)

        self.datasets = {'train': train_loader, 'test': test_loader }

    def get_test_pos_count(self):
        return self.test_dataset.pos_num_of_samples
