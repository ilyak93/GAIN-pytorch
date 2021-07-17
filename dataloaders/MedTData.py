import random
from math import ceil, floor

import PIL.Image
import torch
import os
import numpy as np
from torch.utils import data
from torch.utils.data import SequentialSampler, RandomSampler
from sys import maxsize as maxint

from utils.image import MedT_preprocess_image_v4


def build_balanced_dataloader(dataset, labels, collate_fn, target_weight=None, batch_size=1, steps_per_epoch=500, num_workers=1):
    assert len(dataset) == len(labels)
    labels = np.asarray(labels)
    ulabels, label_count = np.unique(labels, return_counts=True)
    assert (ulabels == list(range(len(ulabels)))).all()
    balancing_weight = 1 / label_count
    target_weight = target_weight if target_weight is not None else np.ones(len(ulabels))
    assert len(target_weight) == len(ulabels)

    from torch.utils.data import WeightedRandomSampler
    num_samples = steps_per_epoch * batch_size
    weighted_sampler = WeightedRandomSampler(
        weights=(target_weight * balancing_weight)[labels],
        num_samples=num_samples,
        replacement=True
    )
    loader = torch.utils.data.DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        sampler=weighted_sampler,
                        collate_fn=collate_fn)
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
    return tensor_image, torch.tensor(-1), label


class MedT_Train_Data(data.Dataset):
    def __init__(self, masks_to_use, mean, std, transform, root_dir='train', loader=load_func):
        self.pos_root_dir = root_dir+'Pos/'
        self.neg_root_dir = root_dir + 'Neg/'
        all_neg_files = os.listdir(self.neg_root_dir)
        all_pos_files = os.listdir(self.pos_root_dir)
        pos_cl_images = [file for file in all_pos_files if 'm' not in file]
        self.masks_indices = [idx for idx,pos in enumerate(pos_cl_images) if pos.split('.')[0]+'m'+'.png' in all_pos_files]
        self.all_files = all_pos_files+all_neg_files
        self.all_cl_images = pos_cl_images+all_neg_files
        self.pos_num_of_samples = len(pos_cl_images)
        self.loader = loader
        mask_max_idx = int(self.pos_num_of_samples * masks_to_use)
        self.used_masks = self.masks_indices[:mask_max_idx]
        self.mean = mean
        self.std = std
        self.transform = transform


    def __len__(self):
        return len(self.all_cl_images)

    def __getitem__(self, index):
        if index < self.pos_num_of_samples:
            res = list(self.loader(self.pos_root_dir,
                                   self.all_cl_images[index], self.all_files))
            preprocessed, augmented, augmented_mask = \
                self.transform(img=res[0].squeeze().permute([2, 0, 1]),
                                         mask=res[1].unsqueeze(0), train=True,
                                         mean=self.mean, std=self.std)
            if index in self.used_masks:
                res = [res[0]] + [preprocessed] + [augmented] + [res[1]]+ \
                      [augmented_mask]+[True] + [res[2]]
            else:
                res = [res[0]] + [preprocessed] + [augmented] + [res[1]] +\
                      [augmented_mask]+[False] + [res[2]]
        else:
            res = list(self.loader(self.neg_root_dir,
                                   self.all_cl_images[index], None))
            preprocessed, augmented, augmented_mask = \
                self.transform(img=res[0].squeeze().permute([2, 0, 1]),
                                         mask=res[1].unsqueeze(0), train=True,
                                         mean=self.mean, std=self.std)
            res = [res[0]] + [preprocessed] + [augmented] + [res[1]] + \
                  [np.array(-1)] + [False] + [res[2]]
        res.append(index)
        return res

    def positive_len(self):
        return self.pos_num_of_samples

    def get_masks_indices(self):
        return self.masks_indices





class MedT_Test_Data(data.Dataset):
    def __init__(self, mean, std, transform, root_dir='validation', loader=load_func):
        self.pos_root_dir = root_dir+'Pos/'
        self.neg_root_dir = root_dir + 'Neg/'
        self.all_files = os.listdir(self.pos_root_dir) + \
                         os.listdir(self.neg_root_dir)
        self.pos_num_of_samples = len(os.listdir(self.pos_root_dir))
        self.loader = loader
        self.mean = mean
        self.std = std
        self.transform = transform

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        if index < self.pos_num_of_samples:
            res = list(self.loader(self.pos_root_dir,
                                   self.all_files[index], None))
        else:
            res = list(self.loader(self.neg_root_dir,
                                   self.all_files[index], None))
        preprocessed, augmented, _ = \
            self.transform(img=res[0].squeeze().numpy(),
                           train=False, mean=self.mean, std=self.std)
        res = [res[0]] + [preprocessed] + [augmented] + [res[1]] + [np.array(-1)] +\
              [False] + [res[2]]
        res.append(index)
        return res

    def positive_len(self):
        return self.pos_num_of_samples




class MedT_Loader():
    def __init__(self, root_dir, target_weight, masks_to_use, mean, std,
                 transform, collate_fn, batch_size=1, steps_per_epoch=6000,
                 num_workers=3):

        self.train_dataset = MedT_Train_Data(root_dir=root_dir+'training/',
                                             masks_to_use=masks_to_use,
                                             mean=mean, std=std,
                                             transform=transform)
        self.test_dataset = MedT_Test_Data(root_dir=root_dir + 'validation/',
                                           mean=mean, std=std,
                                           transform=transform)

        #train_sampler = RandomSampler(self.train_dataset, num_samples=maxint,
        #                              replacement=True)
        test_sampler = SequentialSampler(self.test_dataset)

        train_as_test_sampler = SequentialSampler(self.train_dataset)

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

        train_loader = build_balanced_dataloader(
                    self.train_dataset, labels.int(),
                    target_weight=target_weight, batch_size=batch_size,
                    steps_per_epoch=steps_per_epoch, num_workers=num_workers,
                    collate_fn=collate_fn)

        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            sampler=test_sampler,
            collate_fn=collate_fn)

        train_as_test_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            sampler=train_as_test_sampler)

        self.datasets = {'train': train_loader, 'test': test_loader,
                         'train_as_test': train_as_test_loader }

    def get_test_pos_count(self, train_as_test=False):
        if train_as_test:
            return self.train_dataset.pos_num_of_samples
        return self.test_dataset.pos_num_of_samples

    def get_train_pos_count(self):
        return self.train_dataset.pos_num_of_samples
