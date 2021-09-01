import numpy as np
import os
import torch
from torch.utils.data import Dataset, Subset
from dataloaders.pascal_voc_loader import PascalVOCLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

device_name = 'cpu' # cuda:0
device = torch.device(device_name)

def my_collate(batch):
    images = [torch.tensor(sample['image']) for sample in batch]
    max_len = max(batch, key=lambda p: p['image'].shape[0])['image'].shape[0]
    max_width = max(batch, key=lambda p: p['image'].shape[1])['image'].shape[1]
    labels = [sample['label/idx'] for sample in batch]
    padded = []
    for img in images:
        base = torch.zeros((max_len, max_width)).unsqueeze(2).repeat([1,1,3])
        base[0:img.size()[0], 0:img.size()[1]] = img
        base = base.type(torch.ByteTensor)
        padded.append(base)
    padded_images = torch.stack(padded, dim=0).to(device)


    labels_ohe = [sum(sample['label/onehot']).to(device) for sample in batch]
    return padded_images, labels_ohe, labels


class RawDataset:
    def __init__(self,
                 root_dir,
                 augmentations=None,
                 ds_split=0.8,
                 num_workers=1,
                 output_dims=224,
                 output_channels=3,
                 shuffle=False,
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

        self.seq_train_subset = Subset(self.dataset, train_indices)
        self.seq_test_subset = Subset(self.dataset, val_indices)

        random_train_sampler = SubsetRandomSampler(train_indices)
        random_valid_sampler = SubsetRandomSampler(val_indices)

        random_train_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size['train'],
            num_workers=0,
            sampler=random_train_sampler,
            collate_fn=my_collate)
        random_validation_loader = torch.utils.data.DataLoader(
            self.dataset,
            num_workers=0,
            batch_size=self.batch_size['test'],
            sampler=random_valid_sampler)

        sequential_train_sampler = SequentialSampler(self.seq_train_subset)
        sequential_valid_sampler = SequentialSampler(self.seq_test_subset)

        sequetial_train_loader = torch.utils.data.DataLoader(
            self.seq_train_subset,
            batch_size=self.batch_size['train'],
            num_workers=0,
            sampler=sequential_train_sampler)
        sequetial_validation_loader = torch.utils.data.DataLoader(
            self.seq_test_subset,
            num_workers=0,
            batch_size=self.batch_size['test'],
            sampler=sequential_valid_sampler,
            collate_fn=my_collate)

        self.datasets = {'rnd_train': random_train_loader, 'rnd_test': random_validation_loader,
                         'seq_train': sequetial_train_loader, 'seq_test': sequetial_validation_loader}
