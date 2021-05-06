import random

import PIL.Image
import torch
import os
import numpy as np
from torch.utils import data
from torch.utils.data import SequentialSampler, RandomSampler


def load_func(path, file, all_files, could_be_mask):
    path_to_file = os.path.join(path, file)
    p_image = PIL.Image.open(path_to_file)
    np_image = np.asarray(p_image)
    tensor_image = torch.tensor(np_image)
    img_name, format = str(file).split('.')
    mask_file = img_name+'m'+'.'+format
    if could_be_mask and mask_file in all_files:
        path_to_mask = os.path.join(path, mask_file)
        p_mask = PIL.Image.open(path_to_mask)
        np_mask = np.asarray(p_mask)
        tensor_mask = torch.tensor(np_mask)
        return tensor_image, tensor_mask
    return (tensor_image, -1)


class MedT_Train_Data(data.Dataset):
    def __init__(self, root_dir='train', loader=load_func):
        self.pos_root_dir = root_dir+'Pos/'
        self.neg_root_dir = root_dir + 'Neg/'
        self.all_pos_files = os.listdir(self.pos_root_dir)
        self.all_pos_images = [file for file in self.all_pos_files if 'm' not in file]
        self.all_neg_files = os.listdir(self.neg_root_dir)
        self.pos_num_of_samples = len(self.all_pos_images)
        self.neg_num_of_samples = len(self.all_neg_files)
        self.loader = loader

    def __len__(self):
        return self.pos_num_of_samples+self.neg_num_of_samples

    def __getitem__(self, index):
        if index < self.pos_num_of_samples:
            return self.loader(self.pos_root_dir, self.all_pos_images[index], self.all_pos_files, could_be_mask=True)
        return self.loader(self.neg_root_dir, self.all_neg_files[index - self.pos_num_of_samples], None, could_be_mask=False)


class MedT_Test_Data(data.Dataset):
    def __init__(self, root_dir='train', loader=load_func):
        self.pos_root_dir = root_dir+'Pos/'
        self.neg_root_dir = root_dir + 'Neg/'
        self.all_pos_files = os.listdir(self.pos_root_dir)
        self.all_neg_files = os.listdir(self.neg_root_dir)
        self.pos_num_of_samples = len(self.all_pos_files)
        self.total_num_of_samples = self.pos_num_of_samples + len(self.all_neg_files)
        self.loader = loader

    def __len__(self):
        return self.total_num_of_samples

    def __getitem__(self, index):
        if random.random() <= self.pos_num_of_samples / self.total_num_of_samples:
            return self.loader(self.pos_root_dir, self.all_pos_files[index], self.all_pos_files, could_be_mask=False)

        return self.loader(self.neg_root_dir, self.all_neg_files[index], None, could_be_mask=False)


class MedT_Loader():
    def __init__(self, root_dir):
        self.train_dataset = MedT_Train_Data(root_dir+'training/')
        self.test_dataset = MedT_Test_Data(root_dir + 'validation/')

        train_sampler = RandomSampler(self.train_dataset, replacement=True)
        test_sampler = SequentialSampler(self.test_dataset)

        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=1,
            num_workers=0,
            sampler=train_sampler)

        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            num_workers=0,
            batch_size=1,
            sampler=test_sampler)

        self.datasets = {'train': train_loader, 'test': test_loader }

#loader_1 = DataLoader(ImageData('train/folder_1'), batch_size=3)