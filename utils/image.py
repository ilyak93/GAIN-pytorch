from collections import OrderedDict
from random import choice
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor, \
    AutoAugmentPolicy, RandomHorizontalFlip, RandomRotation, RandomVerticalFlip, RandomPerspective, RandomResizedCrop, \
    Resize
import torchvision

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw


from albumentations import (
    RandomBrightness,
    RandomContrast,
    RandomScale,
    GaussianBlur
)

class RandomRot90:
    """Rotate by [0,90,180,270]."""

    def __init__(self):
        self.choices = list(range(4))

    def __call__(self, img):
        k = random.choice(self.choices)
        if type(img) == Image.Image:
            rotations = [None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
            rotate_config = rotations[k]
            if rotate_config is None:
                img_res = img
            else:
                img_res = img.transpose(rotate_config)
        else:
            img_res = torch.rot90(img, k, [1, 2])

        return img_res

class AddGaussianNoise:
    def __init__(self, mean=0., std=1 / 255):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

#https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py

def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

class ShearTranslate:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = [
            (ShearX, 0., 0.3),
            (ShearY, 0., 0.3),
            (TranslateXabs, 0., 100),
            (TranslateYabs, 0., 100),
        ]

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)

        return img


class MyAugs:
    def __init__(self, n, m):
        self.rotate = random.randint(0,3)
        self.flip = random.randint(0,1)

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)

        return img

def preprocess_image(img , train , mean=None, std=None) -> torch.Tensor:
    if std is None:
        std = [0.5, 0.5, 0.5]
    if mean is None:
        mean = [0.5, 0.5, 0.5]

    #ds_pls = [AutoAugmentPolicy.IMAGENET, AutoAugmentPolicy.CIFAR10, AutoAugmentPolicy.SVHN]
    #random_policy = choice(ds_pls)

    if train == True:
        augment = Compose([
            Image.fromarray,
            torchvision.transforms.AutoAugment(AutoAugmentPolicy.CIFAR10),
            RandomHorizontalFlip(),
        ])
        normilize = Compose([
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
        augmented = augment(img)
        preprocced = normilize(augmented).unsqueeze(0)

        return preprocced, augmented

    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])

    return preprocessing(img).unsqueeze(0), None


def MedT_preprocess_image(img , train , mean=None, std=None) -> torch.Tensor:
    if std is None:
        std = [0.5, 0.5, 0.5]
    if mean is None:
        mean = [0.5, 0.5, 0.5]

    degrees = int(random.random() * 360)

    if train == True:
        augment = Compose([
            Image.fromarray,
            RandomHorizontalFlip(),
            RandomRotation(degrees),
            RandomVerticalFlip(),
        ])
        normilize = Compose([
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
        augmented = augment(img)
        preprocced = normilize(augmented).unsqueeze(0)

        return preprocced, augmented

    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])

    return preprocessing(img).unsqueeze(0), None

def MedT_preprocess_v2_image(img , train , mean=None, std=None) -> torch.Tensor:
    if std is None:
        std = [0.5, 0.5, 0.5]
    if mean is None:
        mean = [0.5, 0.5, 0.5]


    degrees = int(random.random() * 360)
    n, m = random.randint(1, 4), random.randint(2, 15)
    ShearTranslateAug = ShearTranslate(n, m)


    augmentations = [
        RandomHorizontalFlip(),
        RandomRotation(degrees),
        RandomVerticalFlip(),
        RandomPerspective(),
        RandomBrightness(),
        RandomContrast(),
        RandomScale(),
        GaussianBlur(),
        RandomResizedCrop(),
        ShearTranslateAug
    ]

    augs_num_to_apply = random.randint(1, len(augmentations))
    augs = random.sample(augmentations, augs_num_to_apply)

    if train == True:
        augment = Compose([
            Image.fromarray,
            *augs
        ])
        normilize = Compose([
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
        augmented = augment(img)
        preprocced = normilize(augmented).unsqueeze(0)

        return preprocced, augmented

    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])

    return preprocessing(img).unsqueeze(0), None


def MedT_preprocess_image_v3(img , train , mean=None, std=None) -> torch.Tensor:
    if std is None:
        std = [0.5, 0.5, 0.5]
    if mean is None:
        mean = [0.5, 0.5, 0.5]

    if train == True:
        augment = Compose([
            Image.fromarray,
            RandomResizedCrop(224, scale=(0.88, 1.0), ratio=(0.999, 1.001)),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRot90()
        ])
        normilize = Compose([
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
        augmented = augment(img)
        preprocced = normilize(augmented).unsqueeze(0)

        return preprocced, augmented

    preprocessing = Compose([
        Image.fromarray,
        Resize(size=224),
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])

    return preprocessing(img).unsqueeze(0), None



def MedT_preprocess_image_v4(img , train , mask=-1, mean=None, std=None) -> torch.Tensor:
    if std is None:
        std = [0.5, 0.5, 0.5]
    if mean is None:
        mean = [0.5, 0.5, 0.5]

    augmented_mask = np.zeros(1)+(-1)

    if train == True:
        augment = Compose([
            RandomResizedCrop(224, scale=(0.88, 1.0), ratio=(0.999, 1.001)),
            RandomHorizontalFlip(),
            RandomRot90()
        ])
        normilize_augment = Compose([
            ToTensor(),
            AddGaussianNoise(),

        ])
        normilize = Normalize(mean=mean, std=std)

        img_mask = img
        if mask.numel() > 1:
            img_mask = torch.cat((img, mask), dim=0)
        augmented_image_mask = augment(img_mask)
        augmented_image = augmented_image_mask
        if mask.numel() > 1:
            augmented_image = augmented_image_mask[0:3, :, :]
            augmented_mask = np.array(augmented_image_mask.permute([1,2,0]))[:, :, 3]

        augmented_image = augmented_image.permute([1,2,0])
        normilized_and_augmented = normilize_augment(np.array(augmented_image))
        preprocced = normilize(normilized_and_augmented).unsqueeze(0)
        #aug = normilized_and_augmented.clone()
        #mn = aug.min()
        #mx = aug.max()
        #aug -= mn.view(1,1,1)
        #aug /= mx.view(1,1,1)
        #aug = (aug*255).ceil().int().permute([1,2,0])

        return preprocced, augmented_image, augmented_mask

    preprocessing = Compose([
        Image.fromarray,
        Resize(size=224),
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])

    return preprocessing(img).unsqueeze(0), None, augmented_mask


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.min(img)
    import sys
    eps = sys.float_info.epsilon
    img = img / (np.max(img)+eps)

    return np.uint8(img * 255)


def show_cam_on_image(img: np.ndarray, mask: np.ndarray, without_norm : bool) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    if without_norm != True:
        heatmap = np.float32(heatmap) / 255
    cam = 0.5 * heatmap + 0.5 * np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam), heatmap

def denorm(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor