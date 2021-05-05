from random import choice

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor, \
    AutoAugmentPolicy, RandomHorizontalFlip
import torchvision

def preprocess_image(img , train , mean=None, std=None) -> torch.Tensor:
    if std is None:
        std = [0.5, 0.5, 0.5]
    if mean is None:
        mean = [0.5, 0.5, 0.5]

    ds_pls = [AutoAugmentPolicy.IMAGENET, AutoAugmentPolicy.CIFAR10, AutoAugmentPolicy.SVHN]
    random_policy = choice(ds_pls)

    if train == True:
        preprocessing = Compose([
            Image.fromarray,
            torchvision.transforms.AutoAugment(random_policy),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
        preprocessing_ret_augmented = Compose([
            Image.fromarray,
            torchvision.transforms.AutoAugment(random_policy),
            RandomHorizontalFlip(),
        ])

        return preprocessing(img).unsqueeze(0), preprocessing_ret_augmented(img)

    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])

    return preprocessing(img).unsqueeze(0), None


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.min(img)
    img = img / np.max(img)

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