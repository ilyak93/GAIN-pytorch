from random import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.image import denorm


class GAIN(nn.Module):
    def __init__(self, model, grad_layer, num_classes, pretraining_epochs=1, test_first_before_train=False):
        super(GAIN, self).__init__()

        self.model = model

        # print(self.model)
        self.grad_layer = grad_layer

        self.num_classes = num_classes

        # Feed-forward features
        self.feed_forward_features = None
        # Backward features
        self.backward_features = None

        # Register hooks
        self._register_hooks(grad_layer)

        # sigma, omega for making the soft-mask
        self.sigma = 0.25
        self.omega = 10

        self.pretraining_epochs = pretraining_epochs
        self.cur_epoch = 0
        if test_first_before_train == True:
            self.cur_epoch = -1
        self.enable_am = False
        if self.pretraining_epochs == 0:
            self.enable_am = True

    def _register_hooks(self, grad_layer):
        def forward_hook(module, input, output):
            self.feed_forward_features = output

        def backward_hook(module, grad_input, grad_output):
            self.backward_features = grad_output[0]

        gradient_layer_found = False
        for idx, m in self.model.named_modules():
            if idx == grad_layer:
                m.register_forward_hook(forward_hook)
                m.register_backward_hook(backward_hook)
                print("Register forward hook !")
                print("Register backward hook !")
                gradient_layer_found = True
                break

        # for our own sanity, confirm its existence
        if not gradient_layer_found:
            raise AttributeError('Gradient layer %s not found in the internal model' % grad_layer)

    def _to_ohe(self, labels):

        ohe = torch.nn.functional.one_hot(labels, self.num_classes).sum(dim=0).unsqueeze(0).float()
        ohe = torch.autograd.Variable(ohe)

        return ohe

    def _to_ohe_multibatch(self, labels):

        ohe = torch.nn.functional.one_hot(labels, self.num_classes).float()
        ohe = torch.autograd.Variable(ohe)

        return ohe

    def forward(self, images, labels): #TODO: no need for saving the hook results ; Put Nan

        # Remember, only do back-probagation during the training. During the validation, it will be affected by bachnorm
        # dropout, etc. It leads to unstable validation score. It is better to visualize attention maps at the testset

        is_train = self.model.training

        with torch.enable_grad():
            # labels_ohe = self._to_ohe(labels).cuda()
            # labels_ohe.requires_grad = True

            _, _, img_h, img_w = images.size()

            self.model.train(is_train)  # TODO: use is_train
            logits_cl = self.model(images)  # BS x num_classes
            self.model.zero_grad()

            if not is_train:
                pred = F.softmax(logits_cl).argmax(dim=1)
                labels_ohe = self._to_ohe_multibatch(pred).cuda()
            else:
                labels_ohe = torch.stack(labels)

            # gradient = logits * labels_ohe
            grad_logits = (logits_cl * labels_ohe).sum(dim=1)  # BS x num_classes
            grad_logits.backward(retain_graph=True, gradient=torch.ones_like(grad_logits))
            self.model.zero_grad()

        backward_features = self.backward_features  # BS x C x H x W
        fl = self.feed_forward_features  # BS x C x H x W
        weights = F.adaptive_avg_pool2d(backward_features, 1)
        Ac = torch.mul(fl, weights).sum(dim=1, keepdim=True)
        Ac = F.relu(Ac)
        # Ac = F.interpolate(Ac, size=images.size()[2:], mode='bilinear', align_corners=False)
        Ac = F.upsample_bilinear(Ac, size=images.size()[2:])
        heatmap = Ac

        Ac_min = Ac.min()
        Ac_max = Ac.max()
        import sys
        eps = sys.float_info.epsilon
        scaled_ac = (Ac - Ac_min) / (Ac_max - Ac_min + eps)
        mask = F.sigmoid(self.omega * (scaled_ac - self.sigma))
        masked_image = images - images * mask

        # for param in self.model.parameters():
        # param.requires_grad = False

        logits_am = self.model(masked_image)

        # for param in self.model.parameters():
        # param.requires_grad = True

        return logits_cl, logits_am, heatmap, masked_image, mask

    def increase_epoch_count(self):
        self.cur_epoch += 1
        if self.cur_epoch >= self.pretraining_epochs:
            self.enable_am = True

    def AM_enabled(self):
        return self.enable_am
