import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob
from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms
from skimage import transform
import imgaug as iaa
import pandas as pd
from skimage import draw, io
import scipy
import warnings


class PascalVOCLoader(data.Dataset):
    """Data loader for the Pascal VOC semantic segmentation dataset.

    Annotations from both the original VOC data (which consist of RGB images
    in which colours map to specific classes) and the SBD (Berkely) dataset
    (where annotations are stored as .mat files) are converted into a common
    `label_mask` format.  Under this format, each mask is an (M,N) array of
    integer values from 0 to 21, where 0 represents the background class.

    The label masks are stored in a new folder, called `pre_encoded`, which
    is added as a subdirectory of the `SegmentationClass` folder in the
    original Pascal VOC data layout.

    A total of five data splits are provided for working with the VOC data:
        train: The original VOC 2012 training data - 1464 images
        val: The original VOC 2012 validation data - 1449 images
        trainval: The combination of `train` and `val` - 2913 images
        train_aug: The unique images present in both the train split and
                   training images from SBD: - 8829 images (the unique members
                   of the result of combining lists of length 1464 and 8498)
        train_aug_val: The original VOC 2012 validation data minus the images
                   present in `train_aug` (This is done with the same logic as
                   the validation set used in FCN PAMI paper, but with VOC 2012
                   rather than VOC 2011) - 904 images
    """

    def __init__(self,
                 root,
                 augmentations=None,
                 output_dim=224,
                 mode='segmentation'):

        # set mode='classification' to retain images with a single object

        if not ((mode == 'segmentation') | (mode == 'classification')):
            raise Exception(
                'mode must be segmentation or classification, got {}'.format(
                    mode))

        self.sbd_path = os.path.join(root, 'benchmark_RELEASE')
        self.root = os.path.join(root, 'VOCdevkit', 'VOC2012')
        self.augmentations = augmentations
        self.n_classes = 21
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        self.categories = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]

        self.transf_shape = {
            'image': [output_dim] * 2,
            'truth': [output_dim] * 2
        }
        self.transf_normalize = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
        # get all image file names
        self.im_path = pjoin(self.root, "JPEGImages", "*.jpg")
        self.segm_path = pjoin(self.root, "SegmentationClass/pre_encoded")
        self.all_files = sorted(glob.glob(pjoin(self.segm_path, '*.png')))
        self.all_files = [
            os.path.splitext(os.path.split(f)[-1])[0] for f in self.all_files
        ]

        self.setup_annotations()

        # Find images for classification (one object only)
        self.files_semantic = sorted(
            glob.glob(pjoin(self.root, 'ImageSets/Main/*_trainval.txt')))
        self.file_to_cat = dict()
        for f, c in zip(self.files_semantic, self.categories):
            df = pd.read_csv(
                f,
                delim_whitespace=True,
                header=None,
                names=['filename', 'true'])
            self.file_to_cat.update(
                {f_: c
                 for f_ in df[df['true'] == 1]['filename']})

        if (mode == 'classification'):
            self.all_files = [
                f for f in self.all_files if (f in self.file_to_cat.keys())
            ]

        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.compute_circle_masks(radius=0.02)


    @staticmethod
    def get_circle_region_mask(map_, radius=30):
        # get mask of circular shape that contains the most positive pixels
        radius += 1 if radius%2 else 0
        r, c = radius, radius
        shape = (int(2*radius+1), int(2*radius+1))
        rr, cc = draw.circle(r,
                                c,
                                radius,
                                shape=shape)
        filt_mask = np.zeros(shape, dtype=int)
        filt_mask[rr, cc] = 1
        res = scipy.ndimage.convolve(map_, filt_mask)

        pos = np.unravel_index(np.argmax(res), map_.shape)

        mask = np.zeros(map_.shape, dtype=int)
        rr, cc = draw.circle(pos[0], pos[1], radius, shape=map_.shape)
        mask[rr, cc] = 1
        return mask

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):

        truth_path = self.all_files[index]
        im_name = os.path.splitext(os.path.split(truth_path)[-1])[0]
        im_path = pjoin(self.root, "JPEGImages", im_name + ".jpg")

        # path = pjoin(self.root, 'SegmentationClass', 'masks')
        masks_paths = sorted(glob.glob(pjoin(self.root, 'SegmentationClass',
                                      'masks', '{}_*.png'.format(truth_path))))

        masks = [(np.asarray(Image.open(m))[..., 0] > 0).astype(int)
                 for m in masks_paths]
        im = Image.open(pjoin(im_path))
        segm = Image.open(pjoin(self.segm_path, truth_path + ".png"))
        im = np.asarray(im)
        truth = np.asarray(segm)
        truths = [(truth == l).astype(int) for l in np.unique(truth)[1:]]

        classes = [self.categories[l] for l in np.unique(truth)[1:] - 1]
        class_onehot = np.array(
            [[1 if (c == class_) else 0 for c in self.categories]
             for class_ in classes])
        class_idx = [np.nonzero(c_onehot)[0][0] for c_onehot in class_onehot]
        class_onehot = [torch.from_numpy(c_onehot).type(torch.float)
                        for c_onehot in class_onehot]

        out = {
            'image': im,
            'label/truths': truths,
            'label/masks': masks,
            'label/name': classes,
            'label/idx': class_idx,
            'label/onehot': class_onehot
        }

        # apply augmentations
        if (self.augmentations is not None):
            out = self.augmentations(out)

        # normalize and resize
        image = transform.resize(
            out['image'],
            self.transf_shape['image'],
            anti_aliasing=True,
            mode='reflect')

        truths = [
            transform.resize(
                t,
                self.transf_shape['truth'],
                anti_aliasing=True,
                mode='reflect')[np.newaxis, ...] for t in out['label/truths']
        ]

        masks = [
            transform.resize(
                t,
                self.transf_shape['truth'],
                anti_aliasing=True,
                mode='reflect')[np.newaxis, ...] for t in out['label/masks']
        ]

        image = np.array([
            image[..., c] - self.transf_normalize['mean'][c] for c in range(3)
        ])
        image = [
            image[c, ...] / self.transf_normalize['std'][c] for c in range(3)
        ]

        # image = image.transpose((2, 0, 1))
        image = np.array(image)
        out['image'] = torch.from_numpy(np.array(image)).type(torch.float)
        out['label/truths'] = [torch.from_numpy(t).type(torch.float) for t in truths]
        out['label/masks'] = [torch.from_numpy(t).type(torch.float) for t in masks]

        return out


    def sample_uniform(self, n=1):
        ids = np.random.choice(np.arange(0, len(self), size=n, replace=False))

        out = [self.__getitem__(i) for i in ids]
        if (n == 1):
            return out[0]
        else:
            return out

    def get_pascal_labels(self):
        """Load the mapping that associates pascal classes with label colors

        Returns:
            np.ndarray with dimensions (21, 3)
        """
        return np.asarray([
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
        ])

    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes

        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.

        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def decode_segmap(self, label_mask, plot=False):
        """Decode segmentation class labels into a color image

        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.

        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_pascal_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

    def setup_annotations(self):
        """
        Sets up Berkley annotations by adding image indices to the
        `train_aug` split and pre-encode all segmentation labels into the
        common label_mask format (if this has not already been done). This
        function also defines the `train_aug` and `train_aug_val` data splits
        according to the description in the class docstring
        """
        sbd_path = self.sbd_path
        target_path = pjoin(self.root, "SegmentationClass/pre_encoded")
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        path = pjoin(sbd_path, "dataset/train.txt")
        sbd_train_list = tuple(open(path, "r"))
        sbd_train_list = [id_.rstrip() for id_ in sbd_train_list]
        train_aug = self.files["train"] + sbd_train_list

        pre_encoded = glob.glob(pjoin(target_path, "*.png"))

        if len(pre_encoded) != 9733:
            print("Pre-encoding segmentation masks...")
            for ii in tqdm(sbd_train_list):
                lbl_path = pjoin(sbd_path, "dataset/cls", ii + ".mat")
                data = io.loadmat(lbl_path)
                lbl = data["GTcls"][0]["Segmentation"][0].astype(np.int32)
                lbl = m.toimage(lbl, high=lbl.max(), low=lbl.min())
                m.imsave(pjoin(target_path, ii + ".png"), lbl)

    def compute_circle_masks(self, radius):
        path = pjoin(self.root, 'SegmentationClass', 'masks')
        if not os.path.exists(path):
            os.makedirs(path)

            pbar = tqdm(total=len(self.all_files))
            for f in self.all_files:
                truth_path = f
                segm = Image.open(pjoin(self.segm_path, truth_path + ".png"))
                truth = np.asarray(segm)
                truths = [(truth == l).astype(int) for l in np.unique(truth)[1:]]
                shape = np.array(truth.shape)
                for i, t in enumerate(truths):
                    path_out = pjoin(path, '{}_{}.png'.format(f, i))
                    if not (os.path.exists(path_out)):
                        out_filt = PascalVOCLoader.get_circle_region_mask(
                            t,
                            radius=np.max(shape)*radius)[..., np.newaxis]
                        out_filt = (np.repeat(out_filt, 3, axis=-1)*255).astype(np.uint8)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            io.imsave(path_out, out_filt)
                pbar.update(1)
