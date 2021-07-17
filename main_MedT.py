
import pathlib
import math

import argparse
import torch

from torch import nn

from torchvision.models import vgg19, wide_resnet101_2, mobilenet_v2
import numpy as np
import matplotlib.pyplot as plt
# from torchviz import make_dot
from sys import maxsize as maxint
import copy

from torchvision.transforms import Resize, Normalize, ToTensor

from dataloaders import data
from dataloaders.MedTData import MedT_Loader
from metrics.metrics import calc_sensitivity

from models.batch_GAIN_MedT import batch_GAIN_MedT
from utils.image import show_cam_on_image, preprocess_image, deprocess_image, denorm, MedT_preprocess_image, \
    MedT_preprocess_image_v3, MedT_preprocess_image_v4


from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from config import cfg

def my_collate(batch):
    orig_imgs, preprocessed_imgs, agumented_imgs, masks, preprocessed_masks,\
                    used_masks, labels, indices = zip(*batch)
    used_masks = [mask for mask,used in zip(preprocessed_masks, used_masks) if used == True]
    preprocessed_masks = [mask for mask in preprocessed_masks if mask.size > 1]
    res_dict = {'orig_images': orig_imgs,
                'preprocessed_images' : preprocessed_imgs,
                'augmented_images' : agumented_imgs, 'orig_masks': masks,
                'preprocessed_masks' : preprocessed_masks,
                'used_masks' : used_masks,
                'labels': labels, 'idx': indices}
    return res_dict

def monitor_test_epoch(writer, test_dataset, args, pos_count, test_differences,
                       epoch_test_am_loss, test_total_pos_correct, epoch,
                       total_test_single_accuracy, test_total_neg_correct):
    num_test_samples = len(test_dataset)
    print('Average epoch single test accuracy: {:.3f}'.format(total_test_single_accuracy / num_test_samples))

    # epoch_test_multi_accuracy.append(total_test_multi_accuracy / num_test_samples)
    # print('Average epoch multi test accuracy: {:.3f}'.format(total_test_multi_accuracy / num_test_samples))

    writer.add_scalar('Loss/test/am_total_loss', epoch_test_am_loss / (pos_count / args.batchsize), epoch)

    writer.add_scalar('Accuracy/test/cl_accuracy_only_pos',
                      test_total_pos_correct / pos_count, epoch)
    writer.add_scalar('Accuracy/test/cl_accuracy_only_neg',
                      test_total_neg_correct / (num_test_samples - pos_count), epoch)

    writer.add_scalar('Accuracy/test/cl_accuracy', total_test_single_accuracy / num_test_samples, epoch)

    ones = torch.ones(pos_count)
    test_labels = torch.zeros(num_test_samples)
    test_labels[0:len(ones)] = ones
    test_labels = test_labels.int()
    all_sens, auc = calc_sensitivity(test_labels.cpu().numpy(), test_differences)
    writer.add_scalar('ROC/test/ROC_0.1', all_sens[0], epoch)
    writer.add_scalar('ROC/test/ROC_0.05', all_sens[1], epoch)
    writer.add_scalar('ROC/test/AUC', auc, epoch)

def monitor_test_viz(j, t, heatmaps, sample, masked_images, test_dataset,
                       label_idx_list, logits_cl, am_scores, am_labels, writer,
                       epoch, cfg):
    if (j < args.pos_to_write_test and j % t == 0) or (
            j > args.pos_to_write_test and j % (args.pos_to_write_test * 3) == 0):
        htm = np.uint8(heatmaps[0].squeeze().cpu().detach().numpy() * 255)
        resize = Resize(size=224)
        orig = sample['orig_images'][0].permute([2, 0, 1])
        orig = resize(orig).permute([1, 2, 0])
        np_orig = orig.cpu().detach().numpy()
        visualization, heatmap = show_cam_on_image(np_orig, htm, True)
        viz = torch.from_numpy(visualization).unsqueeze(0)
        orig = orig.unsqueeze(0)
        masked_image = denorm(masked_images[0].detach().squeeze(),
                              test_dataset.mean, test_dataset.std)
        masked_image = (masked_image.squeeze().permute([1, 2, 0]).cpu().detach().numpy() * 255).round().astype(
            np.uint8)
        masked_image = torch.from_numpy(masked_image).unsqueeze(0)
        orig_viz = torch.cat((orig, viz, masked_image), 0)
        gt = [cfg['categories'][x] for x in label_idx_list][0]
        writer.add_images(tag='Test_Heatmaps/image_' + str(j) + '_' + gt,
                          img_tensor=orig_viz, dataformats='NHWC', global_step=epoch)
        y_scores = nn.Softmax(dim=1)(logits_cl.detach())
        predicted_categories = y_scores[0].unsqueeze(0).argmax(dim=1)
        predicted_cl = [(cfg['categories'][x], format(y_scores[0].view(-1)[x], '.4f')) for x in
                        predicted_categories.view(-1)]
        labels_cl = [(cfg['categories'][x], format(y_scores[0].view(-1)[x], '.4f')) for x in [(label_idx_list[0])]]
        import itertools
        predicted_cl = list(itertools.chain(*predicted_cl))
        labels_cl = list(itertools.chain(*labels_cl))
        cl_text = 'cl_gt_' + '_'.join(labels_cl) + '_pred_' + '_'.join(predicted_cl)

        predicted_am = [(cfg['categories'][x], format(am_scores[0].view(-1)[x], '.4f')) for x in am_labels[0].view(-1)]
        labels_am = [(cfg['categories'][x], format(am_scores[0].view(-1)[x], '.4f')) for x in [label_idx_list[0]]]
        import itertools
        predicted_am = list(itertools.chain(*predicted_am))
        labels_am = list(itertools.chain(*labels_am))
        am_text = '_am_gt_' + '_'.join(labels_am) + '_pred_' + '_'.join(predicted_am)

        writer.add_text('Test_Heatmaps_Description/image_' + str(j) + '_' + gt, cl_text + am_text,
                        global_step=epoch)

def test(args, cfg, model, device, test_loader, test_dataset, writer, epoch):

    model.eval()

    j = 0
    test_total_pos_correct, test_total_neg_correct = 0, 0
    epoch_test_am_loss, total_test_single_accuracy = 0, 0
    test_differences = np.zeros(len(test_dataset))

    for sample in test_loader:
        label_idx_list = sample['labels']
        batch = torch.stack(sample['preprocessed_images'], dim=0).squeeze()
        batch = batch.to(device)
        labels = torch.Tensor(label_idx_list).to(device).long()

        logits_cl, logits_am, heatmaps, masks, masked_images = model(batch, labels)

        # Single label evaluation
        y_pred = logits_cl.detach().argmax(dim=1)
        y_pred = y_pred.view(-1)
        gt = labels.view(-1)
        acc = (y_pred == gt).sum()
        total_test_single_accuracy += acc.detach().cpu()

        pos_correct = (y_pred == gt).logical_and(gt == 1).sum()
        neg_correct = (y_pred == gt).logical_and(gt == 0).sum()
        test_total_neg_correct += neg_correct
        test_total_pos_correct += pos_correct

        difference = (logits_cl[:, 1] - logits_cl[:, 0]).cpu().detach().numpy()
        test_differences[j * args.batchsize: j * args.batchsize + len(difference)] = difference

        am_scores = nn.Softmax(dim=1)(logits_am)
        am_labels = am_scores.argmax(dim=1)

        pos_indices = [idx for idx, x in enumerate(sample['labels']) if x == 1]
        cur_pos_num = len(pos_indices)
        # The code to replace to train on positives and negatives
        if cur_pos_num > 1:
            am_labels_scores = am_scores[pos_indices, torch.ones(cur_pos_num).long()]
            am_loss = am_labels_scores.sum() / am_labels_scores.size(0)
            epoch_test_am_loss += (am_loss * args.am_weight).detach().cpu().item()

        pos_count = test_dataset.positive_len()
        t = math.ceil(pos_count / (args.batchsize * args.pos_to_write_test))
        monitor_test_viz(j, t, heatmaps, sample, masked_images, test_dataset,
                       label_idx_list, logits_cl, am_scores, am_labels, writer,
                       epoch, cfg)
        j += 1

    monitor_test_epoch(writer, test_dataset, args, pos_count, test_differences,
                       epoch_test_am_loss, test_total_pos_correct, epoch,
                       total_test_single_accuracy, test_total_neg_correct)




def monitor_train_epoch(writer, count_pos, count_neg, epoch, am_count,
                               epoch_train_am_loss, epoch_train_cl_loss,
                               num_train_samples, epoch_train_total_loss,
                               batchsize, epoch_IOU, IOU_count, train_labels,
                               total_train_single_accuracy, test_before_train,
                               train_total_pos_correct, train_total_pos_seen,
                               train_total_neg_correct, train_total_neg_seen,
                               train_differences):
    print("pos = {} neg = {}".format(count_pos, count_neg))
    if (test_before_train and epoch > 0) or test_before_train == False:
        print('Average epoch train am loss: {:.3f}'.format(epoch_train_am_loss
                                                           / am_count))
        print('Average epoch train cl loss: {:.3f}'.format(
            epoch_train_cl_loss / (num_train_samples * batchsize)))
        print('Average epoch train total loss: {:.3f}'.format(
            epoch_train_total_loss / count_pos))
        print('Average epoch single train accuracy: {:.3f}'.format(
            total_train_single_accuracy / (num_train_samples*batchsize)))
    if (test_before_train and epoch > 0) or test_before_train == False:
        writer.add_scalar('Loss/train/cl_total_loss', epoch_train_cl_loss /
                          (num_train_samples * batchsize), epoch)
        writer.add_scalar('Loss/train/am_total_loss', epoch_train_am_loss /
                          am_count, epoch)
        writer.add_scalar('IOU/train/average_IOU_per_sample', epoch_IOU /
                          IOU_count, epoch)
        writer.add_scalar('Accuracy/train/cl_accuracy',
                          total_train_single_accuracy / (num_train_samples *
                                                         batchsize), epoch)
        writer.add_scalar('Accuracy/train/cl_accuracy_only_pos',
                          train_total_pos_correct / train_total_pos_seen,
                          epoch)
        writer.add_scalar('Accuracy/train/cl_accuracy_only_neg',
                          train_total_neg_correct / train_total_neg_seen,
                          epoch)
        all_sens, auc = calc_sensitivity(train_labels, train_differences)
        writer.add_scalar('ROC/train/ROC_0.1', all_sens[0], epoch)
        writer.add_scalar('ROC/train/ROC_0.05', all_sens[1], epoch)
        writer.add_scalar('ROC/train/AUC', auc, epoch)


def monitor_IOU(have_mask_indices, all_augmented_masks, masks, epoch_IOU,
                IOU_count, writer, cfg):
    if len(have_mask_indices) > 0 and cfg['i'] % 100 == 0:
        m1 = torch.tensor(all_augmented_masks).cuda()
        m2 = masks[have_mask_indices].squeeze().round().detach()
        intersection = (m1.logical_and(m2)).int().sum()
        union = (m1.logical_or(m2.squeeze())).int().sum()
        IOU = (intersection / union) / len(have_mask_indices)
        epoch_IOU += IOU
        IOU_count += 1
        writer.add_scalar('IOU/train/', IOU.detach().cpu().item(), cfg['IOU_i'])
        cfg['IOU_i'] += 1
    return epoch_IOU, IOU_count


def monitor_train_iteration(sample, writer, logits_cl, cl_loss,
                            cl_loss_fn, total_loss, epoch, args, cfg, lbs,
                            train_total_pos_seen, train_total_pos_correct,
                            train_total_neg_correct, train_total_neg_seen):
    if cfg['i'] % 100 == 0:
        pos_indices = [idx for idx, x in enumerate(sample['labels']) if x == 1]
        neg_indices = [idx for idx, x in enumerate(sample['labels']) if x == 0]
        pos_correct = len(
            [pos_idx for pos_idx in pos_indices if logits_cl[pos_idx, 1] > logits_cl[pos_idx, 0]])
        neg_correct = len(
            [neg_idx for neg_idx in neg_indices if logits_cl[neg_idx, 1] <= logits_cl[neg_idx, 0]])
        train_total_pos_seen += len(pos_indices)
        train_total_pos_correct += pos_correct
        train_total_neg_correct += neg_correct
        train_total_neg_seen += len(neg_indices)
        if len(pos_indices) > 0:
            cl_loss_only_on_pos_samples = cl_loss_fn(
                logits_cl.detach()[pos_indices], lbs.detach()[pos_indices])
            weighted_cl_pos = cl_loss_only_on_pos_samples * args.cl_weight
            writer.add_scalar('Loss/train/cl_loss_only_on_pos_samples',
                            weighted_cl_pos.detach().cpu().item(), cfg['am_i'])
        cfg['am_i'] += 1
        writer.add_scalar('Loss/train/cl_loss',
                          (cl_loss * args.cl_weight).detach().cpu().item(),
                          cfg['i'])
        writer.add_scalar('Loss/train/total_loss',
                          total_loss.detach().cpu().item(), cfg['total_i'])
        cfg['total_i'] += 1

    cfg['i'] += 1

    if epoch == 0 and args.test_before_train == False:
        cfg['num_train_samples'] += 1
    if epoch == 1 and args.test_before_train == True:
        cfg['num_train_samples'] += 1

    return train_total_pos_seen, train_total_pos_correct,\
           train_total_neg_correct, train_total_neg_seen

def monitor_train_viz(writer, records_indices, heatmaps, augmented_batch,
                      sample, masked_images, train_dataset, label_idx_list,
                      epoch, logits_cl, am_scores, gt, cfg):
    for idx in records_indices:
        htm = np.uint8(heatmaps[idx].squeeze().cpu().detach().numpy() * 255)
        visualization, _ = show_cam_on_image(np.asarray(augmented_batch[idx]), htm, True)
        viz = torch.from_numpy(visualization).unsqueeze(0)
        augmented = torch.tensor(np.asarray(augmented_batch[idx])).unsqueeze(0)
        resize = Resize(size=224)
        orig = sample['orig_images'][idx].permute([2, 0, 1])
        orig = resize(orig).permute([1, 2, 0]).unsqueeze(0)
        masked_img = denorm(masked_images[idx].detach().squeeze(),
                            train_dataset.mean, train_dataset.std)
        masked_img = (masked_img.squeeze().permute([1, 2, 0]).cpu().detach().numpy() * 255).round().astype(
            np.uint8)
        masked_img = torch.from_numpy(masked_img).unsqueeze(0)
        if gt[idx] == 1 and sample['orig_masks'][idx].numel() != 1:
            orig_mask = sample['orig_masks'][idx].unsqueeze(2).repeat([1, 1, 3]).permute([2, 0, 1])
            orig_mask = resize(orig_mask).permute([1, 2, 0])
            orig_mask[orig_mask == 255] = 30
            orig_masked = orig + orig_mask.unsqueeze(0)
            orig_viz = torch.cat((orig, orig_masked, augmented, viz, masked_img), 0)
        else:
            orig_viz = torch.cat((orig, augmented, viz, masked_img), 0)

        groundtruth = [cfg['categories'][x] for x in label_idx_list][idx]
        img_idx = sample['idx'][idx]
        writer.add_images(
            tag='Epoch_' + str(epoch) + '/Train_Heatmaps/image_' + str(img_idx) + '_' + groundtruth,
            img_tensor=orig_viz, dataformats='NHWC', global_step=cfg['counter'][img_idx])
        y_scores = nn.Softmax(dim=1)(logits_cl.detach())
        predicted_categories = y_scores[idx].unsqueeze(0).argmax(dim=1)
        predicted_cl = [(cfg['categories'][x], format(y_scores[idx].view(-1)[x], '.4f')) for x in
                        predicted_categories.view(-1)]
        labels_cl = [(cfg['categories'][x], format(y_scores[idx].view(-1)[x], '.4f')) for x in
                     [(label_idx_list[idx])]]
        import itertools
        predicted_cl = list(itertools.chain(*predicted_cl))
        labels_cl = list(itertools.chain(*labels_cl))
        cl_text = 'cl_gt_' + '_'.join(labels_cl) + '_pred_' + '_'.join(predicted_cl)
        am_labels = am_scores.argmax(dim=1)
        predicted_am = [(cfg['categories'][x], format(am_scores[idx].view(-1)[x], '.4f')) for x in
                        am_labels[idx].view(-1)]
        labels_am = [(cfg['categories'][x], format(am_scores[idx].view(-1)[x], '.4f')) for x in
                     [label_idx_list[idx]]]
        import itertools
        predicted_am = list(itertools.chain(*predicted_am))
        labels_am = list(itertools.chain(*labels_am))
        am_text = '_am_gt_' + '_'.join(labels_am) + '_pred_' + '_'.join(predicted_am)
        writer.add_text('Train_Heatmaps_Description/image_' + str(img_idx) + '_' + groundtruth,
                        cl_text + am_text,
                        global_step=cfg['counter'][img_idx])
        cfg['counter'][img_idx] += 1


def handle_AM_loss(cur_pos_num, am_scores, pos_indices, model, total_loss,
                   epoch_train_am_loss, am_count, writer, cfg, args, labels):
    if not args.am_on_all and cur_pos_num > 1:
        am_labels_scores = am_scores[pos_indices,
                                     torch.ones(cur_pos_num).long()]
        am_loss = am_labels_scores.sum() / am_labels_scores.size(0)
        if model.AM_enabled():
            total_loss += am_loss * args.am_weight
        epoch_train_am_loss += (am_loss * args.am_weight).detach().cpu().item()
        am_count += 1
        if cfg['i'] % 100 == 0:
            writer.add_scalar('Loss/train/am_loss',
                              (am_loss * args.am_weight).detach().cpu().item(),
                              cfg['am_i'])
        return total_loss, epoch_train_am_loss, am_count
    if args.am_on_all:
        am_labels_scores = am_scores[list(range(args.batchsize)), labels]
        am_loss = am_labels_scores.sum() / am_labels_scores.size(0)
        if model.AM_enabled():
            total_loss += am_loss * args.am_weight
        if cfg['i'] % 100 == 0:
            writer.add_scalar('Loss/train/am_loss',
                              (am_loss * args.am_weight).detach().cpu().item(),
                              cfg['am_i'])
        epoch_train_am_loss += (am_loss * args.am_factor).detach().cpu().item()
        am_count += 1
    return total_loss, epoch_train_am_loss, am_count


def handle_EX_loss(model, used_mask_indices, augmented_masks, heatmaps,
                   writer, total_loss, cfg):
    if model.EX_enabled() and len(used_mask_indices) > 0:
        augmented_masks = [ToTensor()(x).cuda() for x in augmented_masks]
        augmented_masks = torch.cat(augmented_masks, dim=0)
        squared_diff = torch.pow(heatmaps[used_mask_indices].squeeze() - augmented_masks, 2)
        flattened_squared_diff = squared_diff.view(len(used_mask_indices), -1)
        flattned_sum = flattened_squared_diff.sum(dim=1)
        flatten_size = flattened_squared_diff.size(1)
        ex_loss = (flattned_sum / flatten_size).sum() / len(used_mask_indices)
        writer.add_scalar('Loss/train/ex_loss',
                          (ex_loss * args.ex_weight).detach().cpu().item(),
                          cfg['ex_i'])
        total_loss += args.ex_weight * ex_loss
        cfg['ex_i'] += 1
    return total_loss

def train(args, cfg, model, device, train_loader, train_dataset, optimizer,
          writer, epoch):
    #switching model to train mode
    model.train()
    #initializing all required variables
    count_pos, count_neg, dif_i, epoch_IOU, am_count = 0, 0, 0, 0, 0
    train_differences = np.zeros(args.epochsize)
    train_labels = np.zeros(args.epochsize)
    total_train_single_accuracy = 0
    epoch_train_cl_loss, epoch_train_am_loss = 0, 0
    epoch_train_total_loss = 0
    IOU_count = 0
    train_total_pos_correct, train_total_pos_seen = 0, 0
    train_total_neg_correct, train_total_neg_seen = 0, 0
    #defining classification loss function
    cl_loss_fn = torch.nn.BCEWithLogitsLoss()
    #data loading loop
    for sample in train_loader:
        #preparing all required data
        label_idx_list = sample['labels']
        augmented_batch = sample['augmented_images']
        augmented_masks = sample['used_masks']
        all_augmented_masks = sample['preprocessed_masks']
        batch = torch.stack(sample['preprocessed_images'], dim=0).squeeze()
        batch = batch.to(device)
        #starting the forward, backward, optimzer.step process
        optimizer.zero_grad()
        labels = torch.Tensor(label_idx_list).to(device).long()
        #sanity check for batch pos and neg distribution (for printing) #TODO: can be removed as it checked and it is ok
        count_pos += (labels == 1).int().sum()
        count_neg += (labels == 0).int().sum()
        #one_hot transformation
        lb1 = labels.unsqueeze(0)
        lb2 = 1 - lb1
        lbs = torch.cat((lb2, lb1), dim=0).transpose(0, 1).float()
        #model forward
        logits_cl, logits_am, heatmaps, masks, masked_images = \
            model(batch, lbs)
        #cl_loss and total loss computation
        cl_loss = cl_loss_fn(logits_cl, lbs)
        total_loss = 0
        total_loss += cl_loss * args.cl_weight
        # AM loss computation and monitoring
        pos_indices = [idx for idx, x in enumerate(sample['labels']) if x == 1]
        cur_pos_num = len(pos_indices)
        am_scores = nn.Softmax(dim=1)(logits_am)
        total_loss, epoch_train_am_loss, am_count = handle_AM_loss(
            cur_pos_num, am_scores, pos_indices, model, total_loss,
            epoch_train_am_loss, am_count, writer, cfg, args, labels)
        #monitoring cl_loss per epoch
        epoch_train_cl_loss += (cl_loss * args.cl_weight).detach().cpu().item()
        #saving logits difference for ROC monitoring
        difference = (logits_cl[:, 1] - logits_cl[:, 0]).cpu().detach().numpy()
        train_differences[dif_i: dif_i + len(difference)] = difference
        train_labels[dif_i: dif_i + len(difference)] = labels.squeeze().cpu().detach().numpy()
        dif_i += len(difference)
        # IOU monitoring
        all_masks = train_dataset.get_masks_indices()
        have_mask_indices = [sample['idx'].index(x) for x in sample['idx']
                             if x in all_masks]
        epoch_IOU, IOU_count = monitor_IOU(
            have_mask_indices, all_augmented_masks, masks, epoch_IOU,
            IOU_count, writer, cfg)
        #Ex loss computation and monitoring
        used_mask_indices = [sample['idx'].index(x) for x in sample['idx']
                             if x in train_dataset.used_masks]
        total_loss = handle_EX_loss(model, used_mask_indices, augmented_masks,
                                    heatmaps, writer, total_loss, cfg)
        #optimization
        total_loss.backward()
        optimizer.step()
        # Single label evaluation
        y_pred = logits_cl.detach().argmax(dim=1)
        y_pred = y_pred.view(-1)
        gt = labels.view(-1)
        acc = (y_pred == gt).sum()
        total_train_single_accuracy += acc.detach().cpu()
        #monitoring per iteration measurements
        train_total_pos_seen, train_total_pos_correct, \
        train_total_neg_correct,train_total_neg_seen = \
            monitor_train_iteration(
                sample, writer, logits_cl, cl_loss, cl_loss_fn,
                total_loss, epoch, args, cfg, lbs, train_total_pos_seen,
                train_total_pos_correct, train_total_neg_correct,
                train_total_neg_seen)
        #monitoring visualizations which were choosen for recording
        pos_neg = cfg['counter'].keys()
        records_indices = [sample['idx'].index(x) for x in sample['idx'] if x in pos_neg]
        monitor_train_viz(writer, records_indices, heatmaps, augmented_batch,
                          sample, masked_images, train_dataset, label_idx_list,
                          epoch, logits_cl, am_scores, gt, cfg)
    #monitoring per epoch measurements
    monitor_train_epoch(
        writer, count_pos, count_neg, epoch, am_count, epoch_train_am_loss,
        epoch_train_cl_loss, cfg['num_train_samples'],
        epoch_train_total_loss, args.batchsize, epoch_IOU, IOU_count,
        train_labels, total_train_single_accuracy, args.test_before_train,
        train_total_pos_correct, train_total_pos_seen,
        train_total_neg_correct, train_total_neg_seen, train_differences)



# Parse all the input argument
parser = argparse.ArgumentParser(description='PyTorch GAIN Training')
parser.add_argument('--batchsize', type=int, default=cfg.BATCHSIZE, help='batch size')
parser.add_argument('--total_epochs', type=int, default=35, help='total number of epoch to train')
parser.add_argument('--nepoch', type=int, default=6000, help='number of iterations per epoch')
parser.add_argument('--nepoch_am', type=int, default=100, help='number of epochs to train without am loss')
parser.add_argument('--nepoch_ex', type=int, default=1, help='number of epochs to train without ex loss')
parser.add_argument('--masks_to_use', type=float, default=0.1, help='the relative number of masks to use in ex-supevision training')

parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--manualSeed', default=cfg.RNG_SEED, type=int, help='manual seed')
parser.add_argument('--net', dest='torchmodel', help='path to the pretrained weights file', default=None, type=str)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--level', default=0, type=int, help='epoch to resume from')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--deviceID', type=int, help='deviceID', default=0)
parser.add_argument('--num_workers', type=int, help='workers number for the dataloaders', default=3)

parser.add_argument('--tensorboard', help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--pos_to_write_train', type=int, help='train positive samples visualizations to monitor in tb', default=50)
parser.add_argument('--neg_to_write_train', type=int, help='train negative samples visualizations to monitor in tb', default=20)
parser.add_argument('--pos_to_write_test', type=int, help='test positive samples visualizations to monitor in tb', default=50)
parser.add_argument('--neg_to_write_test', type=int, help='test negative samples visualizations to monitor in tb', default=20)
parser.add_argument('--log_name', type=str, help='identifying name for storing tensorboard logs')
parser.add_argument('--test_before_train', type=int, default=0, help='test before train epoch')
parser.add_argument('--batch_pos_dist', type=float, help='positive relative amount in a batch', default=0.25)
parser.add_argument('--fill_color', type=list, help='fill color of masked area in AM training', default=[0.4948,0.3301,0.16])
parser.add_argument('--grad_layer', help='path to the input idr', type=str, default='features')
parser.add_argument('--cl_weight', default=1, type=int, help='classification loss weight')
parser.add_argument('--am_weight', default=1, type=int, help='attention-mining loss weight')
parser.add_argument('--ex_weight', default=1, type=int, help='extra-supervision loss weight')
parser.add_argument('--am_on_all', default=0, type=int, help='train am on positives and negatives')

parser.add_argument('--input_dir', help='path to the input idr', type=str)
parser.add_argument('--output_dir', help='path to the outputdir', type=str)
parser.add_argument('--checkpoint_name', help='checkpoint name', type=str)



def main(args):
    categories = [
        'Neg', 'Pos'
    ]

    num_classes = len(categories)
    device = torch.device('cuda:'+str(args.deviceID))
    model = mobilenet_v2(pretrained=True).train().to(device)

    # change the last layer for finetuning
    classifier = model.classifier
    num_ftrs = classifier[-1].in_features
    new_classifier = torch.nn.Sequential(*(list(model.classifier.children())[:-1]),
                                         nn.Linear(num_ftrs, num_classes).to(device))
    model.classifier = new_classifier
    model.train()
    # target_layer = model.features[-1]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    batch_size = args.batchsize
    epoch_size = args.nepoch
    medt_loader = MedT_Loader(args.input_dir,[1-args.batch_pos_dist, args.batch_pos_dist],
                              batch_size=batch_size, steps_per_epoch=epoch_size,
                              masks_to_use=args.masks_to_use, mean=mean, std=std,
                              transform=MedT_preprocess_image_v4,
                              collate_fn=my_collate)

    #if True test epoch will run first
    test_first_before_train = bool(args.test_before_train)


    epochs = args.total_epochs

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    norm = Normalize(mean=mean, std=std)
    fill_color = norm(torch.tensor(args.fill_color).view(1,3,1,1)).cuda()
    model = batch_GAIN_MedT(model=model, grad_layer=args.grad_layer, num_classes=num_classes,
                         am_pretraining_epochs=args.nepoch_am,
                         ex_pretraining_epochs=args.nepoch_ex,
                         fill_color=fill_color,
                         test_first_before_train=test_first_before_train)

    chkpnt_epoch = 0

    #checkpoint = torch.load('C:\Users\Student1\PycharmProjects\GCAM\checkpoints\batch_GAIN\with_am_no_ex_1_')
    #model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #chkpnt_epoch = checkpoint['epoch']+1
    #gain.cur_epoch = chkpnt_epoch
    #if gain.cur_epoch > gain.am_pretraining_epochs:
    #    gain.enable_am = True
    #if gain.cur_epoch > gain.ex_pretraining_epochs:
    #    gain.enable_ex = True

    writer = SummaryWriter(args.output_dir + args.log_name +'_'+
                           datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    i = 0
    num_train_samples = 0
    args.epochsize = epoch_size * batch_size
    am_i = 0
    ex_i = 0
    total_i = 0
    IOU_i = 0

    pos_to_write = args.pos_to_write_train
    neg_to_write = args.neg_to_write_train
    pos_idx = list(range(pos_to_write))
    pos_count = medt_loader.get_train_pos_count()
    neg_idx = list(range(pos_count, pos_count+neg_to_write))
    idx = pos_idx+neg_idx
    counter = dict({x: 0 for x in idx})



    cfg = {'categories' : categories, 'i' : i, 'num_train_samples' : num_train_samples,
           'am_i' : am_i, 'ex_i' : ex_i, 'total_i' : total_i,
           'IOU_i' : IOU_i, 'counter':counter}



    for epoch in range(chkpnt_epoch, epochs):
        if not test_first_before_train or \
                (test_first_before_train and epoch != 0):
            train(args, cfg, model, device, medt_loader.datasets['train'],
                  medt_loader.train_dataset, optimizer, writer, epoch)

        test(args, cfg, model, device, medt_loader.datasets['test'],
                  medt_loader.test_dataset, writer, epoch)

        print("finished epoch number:")
        print(epoch)

        model.increase_epoch_count()

        chkpt_path = args.output_dir+'/checkpoints/'
        pathlib.Path(chkpt_path).mkdir(parents=True, exist_ok=True)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
         }, chkpt_path + args.checkpoint_name+datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)