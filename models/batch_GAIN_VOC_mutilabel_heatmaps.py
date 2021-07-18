from random import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.image import denorm

def is_bn(m):
    return isinstance(m, nn.modules.batchnorm.BatchNorm2d) | isinstance(m, nn.modules.batchnorm.BatchNorm1d)

def take_bn_layers(model):
    for m in model.modules():
        if is_bn(m):
            yield m

class FreezedBnModel(nn.Module):
    def __init__(self, model, is_train=True):
        super(FreezedBnModel, self).__init__()
        self.model = model
        self.bn_layers = list(take_bn_layers(self.model))


    def forward(self, x):
        is_train = self.bn_layers[0].training
        if is_train:
            self.set_bn_train_status(is_train=False)
        predicted = self.model(x)
        if is_train:
            self.set_bn_train_status(is_train=True)

        return predicted

    def set_bn_train_status(self, is_train: bool):
        for layer in self.bn_layers:
            layer.train(mode=is_train)
            layer.weight.requires_grad = is_train #TODO: layer.requires_grad = is_train - check is its OK
            layer.bias.requires_grad = is_train


class batch_GAIN_VOC_multiheatmaps(nn.Module):
    def __init__(self, model, grad_layer, num_classes, batchsize,
                 pretraining_epochs=1, test_first_before_train=False):

        super(batch_GAIN_VOC_multiheatmaps, self).__init__()

        self.model = model

        self.freezed_bn_model = FreezedBnModel(model)

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
        self.sigma = 0.5
        self.omega = 10

        self.pretraining_epochs = pretraining_epochs
        self.cur_epoch = 0
        if test_first_before_train == True:
            self.cur_epoch = -1
        self.enable_am = False
        if self.pretraining_epochs == 0:
            self.enable_am = True

        self.batchsize = batchsize

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
                if type(labels) is tuple or type(labels) is list:
                    labels_ohe = torch.stack(labels)
                else:
                    labels_ohe = labels

            multilabel_not_in_batch = (labels_ohe.sum(1) > torch.ones_like(labels_ohe.sum(1))).int().sum() == 0
            if multilabel_not_in_batch:
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

                Ac_count = Ac.size(0)
                Ac_min, _ = Ac.view(Ac_count, -1).min(dim=1)
                Ac_max, _ = Ac.view(Ac_count, -1).max(dim=1)
                import sys
                eps = torch.tensor(sys.float_info.epsilon).cuda()
                scaled_ac = (Ac - Ac_min.view(-1, 1, 1, 1)) / \
                            (Ac_max.view(-1, 1, 1, 1) - Ac_min.view(-1, 1, 1, 1)
                             + eps.view(1, 1, 1, 1))
                mask = F.sigmoid(self.omega * (scaled_ac - self.sigma))
                masked_image = images - images * mask

                logits_am = self.freezed_bn_model(masked_image)

                return logits_cl, logits_am, heatmap, masked_image, mask

            heatmap_list = []

            max_amount_of_heatmaps_per_sample = labels_ohe.sum(dim=1).max()
            for j in range(labels_ohe.size(1)):
                cur_ohe = torch.zeros(labels_ohe.size()).cuda()
                if labels_ohe[0:labels_ohe.size(0), j].sum() == 0:
                    continue
                cur_ohe[0:labels_ohe.size(0), j] = labels_ohe[0:labels_ohe.size(0), j]
                non_zero_indices = labels_ohe[0:labels_ohe.size(0), j].nonzero(as_tuple=True)
                grad_logits = (logits_cl * cur_ohe).sum(dim=1)  # BS x num_classes
                grad = torch.zeros_like(grad_logits)
                grad[non_zero_indices] = 1
                grad_logits.backward(retain_graph=True, gradient=grad)
                self.model.zero_grad()

                backward_features = self.backward_features  # BS x C x H x W
                fl = self.feed_forward_features  # BS x C x H x W
                weights = F.adaptive_avg_pool2d(backward_features, 1)
                Ac = torch.mul(fl, weights).sum(dim=1, keepdim=True)
                Ac = F.relu(Ac)
                # Ac = F.interpolate(Ac, size=images.size()[2:], mode='bilinear', align_corners=False)
                Ac = F.upsample_bilinear(Ac, size=images.size()[2:])
                heatmap_list += [Ac]

        Ac = torch.cat(heatmap_list, 1)
        d = {}
        tmp_v = labels_ohe.nonzero(as_tuple=False)
        tmp_v2 = tmp_v[:, 1]
        tmp_v3 = tmp_v2.unique().sort()
        d = dict(zip(tmp_v3[0].tolist(), tmp_v3[1].tolist()))
        tmp_v = tmp_v.tolist()
        tmp_v = [[t[0], d[t[1]]]for t in tmp_v]
        ll = [[] for _ in range(len(tmp_v))]
        [ll[t[0]].append(t[1]) for t in tmp_v]
        indices = [l for l in ll if len(l) > 0]

        new_Ac = [Ac[i, idx] for i, idx in enumerate(indices)]
        Ac = []

        for t in new_Ac:
            if t.size(0) < max_amount_of_heatmaps_per_sample:
                Ac.append(torch.cat((t, t[0].repeat((max_amount_of_heatmaps_per_sample - t.size(0)).int().item(), 1, 1))))
            else:
                Ac.append(t)

        Ac = torch.cat(Ac, dim=0).unsqueeze(1)

        heatmap = Ac
        Ac_count = Ac.size(0)

        Ac_min, _ = Ac.view(Ac_count, -1).min(dim=1)
        Ac_max, _ = Ac.view(Ac_count, -1).max(dim=1)
        import sys
        eps = torch.tensor(sys.float_info.epsilon).cuda()
        scaled_ac = (Ac - Ac_min.view(-1, 1, 1, 1)) / \
                    (Ac_max.view(-1, 1, 1, 1) - Ac_min.view(-1, 1, 1, 1)
                     + eps.view(1, 1, 1, 1))
        mask = F.sigmoid(self.omega * (scaled_ac - self.sigma))


        amount_of_heatmaps_per_sample = labels_ohe.sum(dim=1)
        images_with_copies = []
        for j in range(images.size(0)):
            repeated_j_img = images[j].unsqueeze(0).repeat(max_amount_of_heatmaps_per_sample.int().item(), 1, 1, 1)
            images_with_copies.append(repeated_j_img)
        images_with_copies = torch.cat(images_with_copies, dim=0)
        masked_image = images_with_copies - images_with_copies * mask

        # for param in self.model.parameters():
        # param.requires_grad = False
        shape = masked_image.size()
        masked_image = masked_image.reshape(-1, max_amount_of_heatmaps_per_sample.int().item(), shape[1], shape[2], shape[3]).transpose(0,1).reshape(shape)
        batches = list(torch.split(masked_image, split_size_or_sections=self.batchsize, dim=0))

        final_logits_am = 0
        l = amount_of_heatmaps_per_sample.nonzero(as_tuple=True)
        aux = torch.zeros_like(l[0]).cuda()
        for i in range(len(batches)):
            cur_logits_am = self.freezed_bn_model(batches[i])
            tmp = torch.zeros_like(cur_logits_am)
            non_zero_indices = amount_of_heatmaps_per_sample.nonzero(as_tuple=True)
            tmp[non_zero_indices] += cur_logits_am[non_zero_indices]
            final_logits_am += tmp
            aux[non_zero_indices] += 1
            amount_of_heatmaps_per_sample -= 1
            amount_of_heatmaps_per_sample[amount_of_heatmaps_per_sample < 0] = 0
        final_logits_am /= aux.unsqueeze(1)

        # for param in self.model.parameters():
        # param.requires_grad = True

        return logits_cl, final_logits_am, heatmap, masked_image, mask

    def increase_epoch_count(self):
        self.cur_epoch += 1
        if self.cur_epoch >= self.pretraining_epochs:
            self.enable_am = True

    def AM_enabled(self):
        return self.enable_am
