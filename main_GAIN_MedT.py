import PIL.Image
import pathlib
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
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
     MedT_preprocess_image_v4

from PIL import Image

from torch.utils.tensorboard import SummaryWriter
import datetime


def main():
    categories = [
        'Neg', 'Pos'
    ]

    num_classes = len(categories)
    device = torch.device('cuda:0')
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

    batch_size = 24
    epoch_size = 6000
    medt_loader = MedT_Loader('C:/MDT_dataset/SB3_ulcers_mined_roi_mult/',
                              [0.75, 0.25], batch_size=batch_size,
                              steps_per_epoch=epoch_size)

    test_first_before_train = False

    epochs = 100
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    norm = Normalize(mean=mean, std=std)
    fill_color = norm(torch.tensor([0.4948, 0.3301, 0.16]).view(1, 3, 1, 1)).cuda()
    gain = batch_GAIN_MedT(model=model, grad_layer='features', num_classes=2,
                         am_pretraining_epochs=100, ex_pretraining_epochs=1,
                         fill_color=fill_color,
                         test_first_before_train=test_first_before_train)
    cl_factor = 1
    ex_factor = 1
    am_factor = 1

    chkpnt_epoch = 0

    # checkpoint = torch.load('C:\Users\Student1\PycharmProjects\GCAM\checkpoints\batch_GAIN_MedT\with_am_no_ex_1_')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # chkpnt_epoch = checkpoint['epoch']+1
    # gain.cur_epoch = chkpnt_epoch
    # if gain.cur_epoch > gain.am_pretraining_epochs:
    #    gain.enable_am = True
    # if gain.cur_epoch > gain.ex_pretraining_epochs:
    #    gain.enable_ex = True

    writer = SummaryWriter(
        "C:/Users/Student1/PycharmProjects/GCAM" + "/MedT_final_cl_gain_e_6000_b_24_v3_only_pos_am_with_ex_pretrain_2_1_sigma_0.7_omega_30_grad_no_color_weight_"
        + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    i = 0
    num_train_samples = 0
    epoch_size = epoch_size * batch_size
    am_i = 0
    ex_i = 0
    total_i = 0
    IOU_i = 0

    pos_to_write = 50
    neg_to_write = 20
    pos_idx = list(range(pos_to_write))
    pos_count = medt_loader.get_train_pos_count()
    neg_idx = list(range(pos_count, pos_count + neg_to_write))
    idx = pos_idx + neg_idx
    counter = dict({x: 0 for x in idx})

    # masks_counter = dict()
    # masks_count = medt_loader.get_train_pos_count()

    masks_indices = medt_loader.train_dataset.get_masks_indices()

    pos_count = medt_loader.get_train_pos_count()

    masks_to_use = int(pos_count * 0.05)

    all_masks = masks_indices

    used_masks = masks_indices[:masks_to_use]

    for epoch in range(chkpnt_epoch, epochs):
        count_pos = 0
        count_neg = 0
        train_differences = np.zeros(epoch_size * batch_size)
        train_labels = np.zeros(epoch_size * batch_size)
        dif_i = 0

        total_train_single_accuracy = 0
        total_test_single_accuracy = 0

        epoch_train_am_loss = 0
        epoch_train_cl_loss = 0
        epoch_train_total_loss = 0

        model.train(True)

        epoch_IOU = 0

        IOU_prev = IOU_i
        am_count = 0
        train_total_pos_correct = 0
        train_total_pos_seen = 0
        train_total_neg_correct = 0
        train_total_neg_seen = 0
        if not test_first_before_train or (test_first_before_train and epoch != 0):
            for sample in medt_loader.datasets['train']:

                label_idx_list = sample['labels']
                augmented_batch = []
                augmented_masks = []
                all_augmented_masks = []
                batch, augmented, augmented_mask = \
                    MedT_preprocess_image_v4(
                        img=sample['images'][0].squeeze().permute([2, 0, 1]),
                        mask=sample['masks'][0].unsqueeze(0),
                        train=True, mean=mean, std=std)
                augmented_batch.append(augmented)
                if augmented_mask.size > 1:
                    all_augmented_masks.append(augmented_mask)
                    if sample['idx'][0] in used_masks:
                        augmented_masks.append(augmented_mask)
                kk = 1
                for img, mask in zip(sample['images'][1:], sample['masks'][1:]):
                    input_tensor, augmented, augmented_mask = \
                        MedT_preprocess_image_v4(img=img.squeeze().permute([2, 0, 1]),
                                                 mask=mask.unsqueeze(0), train=True,
                                                 mean=mean, std=std)
                    batch = torch.cat((batch, input_tensor), dim=0)
                    augmented_batch.append(augmented)
                    if augmented_mask.size > 1:
                        all_augmented_masks.append(augmented_mask)
                        if sample['idx'][kk] in used_masks:
                            augmented_masks.append(augmented_mask)
                    kk += 1

                batch = batch.to(device)

                optimizer.zero_grad()
                labels = torch.Tensor(label_idx_list).to(device).long()

                count_pos += (labels == 1).int().sum()
                count_neg += (labels == 0).int().sum()

                # logits_cl = model(batch)
                lb1 = labels.unsqueeze(0)
                lb2 = 1 - lb1
                lbs = torch.cat((lb2, lb1), dim=0).transpose(0, 1).float()

                logits_cl, logits_am, heatmaps, masks, masked_images = gain(batch, lbs)

                # indices = torch.Tensor(label_idx_list).long().to(device)
                # class_onehot = torch.nn.functional.one_hot(indices, num_classes).sum(dim=0).unsqueeze(0).float()

                # g = make_dot(am_loss, dict(gain.named_parameters()), show_attrs = True, show_saved = True)
                # g.save('grad_viz', train_path)

                cl_loss = loss_fn(logits_cl, lbs)

                total_loss = 0

                total_loss += cl_loss

                pos_indices = [idx for idx, x in enumerate(sample['labels']) if x == 1]
                cur_pos_num = len(pos_indices)
                am_scores = nn.Softmax(dim=1)(logits_am)
                if cur_pos_num > 1:
                    am_labels_scores = am_scores[pos_indices, torch.ones(cur_pos_num).long()]
                    am_loss = am_labels_scores.sum() / am_labels_scores.size(0)
                    if gain.AM_enabled():
                        total_loss += am_loss * am_factor

                    epoch_train_am_loss += (am_loss * am_factor).detach().cpu().item()
                    am_count += 1

                if i % 100 == 0:
                    writer.add_scalar('Loss/train/am_loss', (am_loss * am_factor).detach().cpu().item(), am_i)
                    pos_indices = [idx for idx, x in enumerate(sample['labels']) if x == 1]
                    neg_indices = [idx for idx, x in enumerate(sample['labels']) if x == 0]
                    pos_correct = len([pos_idx for pos_idx in pos_indices if logits_cl[pos_idx, 1] > logits_cl[pos_idx, 0]])
                    neg_correct = len([neg_idx for neg_idx in neg_indices if logits_cl[neg_idx, 1] <= logits_cl[neg_idx, 0]])
                    train_total_pos_seen += len(pos_indices)
                    train_total_pos_correct += pos_correct
                    train_total_neg_correct += neg_correct
                    train_total_neg_seen += len(neg_indices)
                    cl_loss_only_on_am_samples = loss_fn(logits_cl.detach()[pos_indices], lbs.detach()[pos_indices])
                    writer.add_scalar('Loss/train/cl_loss_only_on_pos_samples',
                                      (cl_loss_only_on_am_samples * cl_factor).detach().cpu().item(), am_i)
                    am_i += 1
                    writer.add_scalar('Loss/train/cl_loss', (cl_loss * cl_factor).detach().cpu().item(), i)

                epoch_train_cl_loss += (cl_loss * cl_factor).detach().cpu().item()

                difference = (logits_cl[:, 1] - logits_cl[:, 0]).cpu().detach().numpy()
                train_differences[dif_i: dif_i + len(difference)] = difference
                train_labels[dif_i: dif_i + len(difference)] = labels.squeeze().cpu().detach().numpy()
                dif_i += len(difference)

                # IOU monitoring
                have_mask_indices = [sample['idx'].index(x) for x in sample['idx'] if x in all_masks]
                if len(have_mask_indices) > 0 and i % 100 == 0:
                    m1 = torch.tensor(all_augmented_masks).cuda()
                    m2 = masks[have_mask_indices].squeeze().round().detach()

                    intersection = (m1.logical_and(m2)).int().sum()
                    union = (m1.logical_or(m2.squeeze())).int().sum()

                    IOU = (intersection / union) / len(have_mask_indices)

                    epoch_IOU += IOU

                    writer.add_scalar('IOU/train/', IOU.detach().cpu().item(), IOU_i)

                    IOU_i += 1

                used_mask_indices = [sample['idx'].index(x) for x in sample['idx'] if x in used_masks]
                if gain.EX_enabled() and len(used_mask_indices) > 0:
                    augmented_masks = [ToTensor()(x).cuda() for x in augmented_masks]
                    augmented_masks = torch.cat(augmented_masks, dim=0)
                    squared_diff = torch.pow(heatmaps[used_mask_indices].squeeze() - augmented_masks, 2)
                    flattened_squared_diff = squared_diff.view(len(used_mask_indices), -1)
                    flattned_sum = flattened_squared_diff.sum(dim=1)
                    flatten_size = flattened_squared_diff.size(1)
                    ex_loss = (flattned_sum / flatten_size).sum() / len(used_mask_indices)
                    writer.add_scalar('Loss/train/ex_loss', (ex_loss * ex_factor).detach().cpu().item(), ex_i)
                    total_loss += ex_factor * ex_loss
                    ex_i += 1

                if i % 100 == 0:
                    writer.add_scalar('Loss/train/total_loss', total_loss.detach().cpu().item(), total_i)
                    total_i += 1

                total_loss.backward()
                optimizer.step()

                # Single label evaluation
                y_pred = logits_cl.detach().argmax(dim=1)
                y_pred = y_pred.view(-1)
                gt = labels.view(-1)
                acc = (y_pred == gt).sum()
                total_train_single_accuracy += acc.detach().cpu()
                i += 1
                if epoch == 0 and test_first_before_train == False:
                    num_train_samples += 1
                if epoch == 1 and test_first_before_train == True:
                    num_train_samples += 1

                # Multi label evaluation
                # _, y_pred_multi = logits_cl.detach().topk(num_of_labels)
                # y_pred_multi = y_pred_multi.view(-1)
                # acc_multi = (y_pred_multi == gt).sum() / num_of_labels
                # total_train_multi_accuracy += acc_multi.detach().cpu()

                pos_neg = counter.keys()
                records_indices = [sample['idx'].index(x) for x in sample['idx'] if x in pos_neg]

                for idx in records_indices:
                    htm = np.uint8(heatmaps[idx].squeeze().cpu().detach().numpy() * 255)
                    visualization, _ = show_cam_on_image(np.asarray(augmented_batch[idx]), htm, True)
                    viz = torch.from_numpy(visualization).unsqueeze(0)
                    augmented = torch.tensor(np.asarray(augmented_batch[idx])).unsqueeze(0)
                    resize = Resize(size=224)
                    orig = sample['images'][idx].permute([2, 0, 1])
                    orig = resize(orig).permute([1, 2, 0]).unsqueeze(0)
                    masked_img = denorm(masked_images[idx].detach().squeeze(), mean, std)
                    masked_img = (masked_img.squeeze().permute([1, 2, 0]).cpu().detach().numpy() * 255).round().astype(
                        np.uint8)
                    masked_img = torch.from_numpy(masked_img).unsqueeze(0)
                    if gt[idx] == 1 and sample['masks'][idx].numel() != 1:
                        orig_mask = sample['masks'][idx].unsqueeze(2).repeat([1, 1, 3]).permute([2, 0, 1])
                        orig_mask = resize(orig_mask).permute([1, 2, 0])
                        orig_mask[orig_mask == 255] = 30
                        orig_masked = orig + orig_mask.unsqueeze(0)
                        orig_viz = torch.cat((orig, orig_masked, augmented, viz, masked_img), 0)
                        '''
                        augmented_orig_mask_tmp = augmented_masks_with_nones[idx]
                        mx_tmp = augmented_orig_mask_tmp.max()
                        augmented_orig_mask_tmp[augmented_orig_mask_tmp == mx_tmp] = mx_tmp / 10
                        orig_masked_tmp = batch[idx] + augmented_orig_mask_tmp
                        orig_masked_tmp = orig_masked_tmp.permute([1, 2, 0]).unsqueeze(0).cpu()
                        orig_viz = torch.cat((orig, orig_masked_tmp, augmented, viz, masked_img), 0)
                        plt.imshow(orig_masked_tmp.squeeze())
                        plt.show()

                        plt.imshow(orig.squeeze())
                        plt.show()

                        plt.imshow(orig_mask)
                        plt.show()

                        plt.imshow(augmented_batch[idx].squeeze())
                        plt.show()

                        plt.imshow(augmented_masks_with_nones[idx])
                        plt.show()
                        '''
                    else:
                        orig_viz = torch.cat((orig, augmented, viz, masked_img), 0)

                    groundtruth = [categories[x] for x in label_idx_list][idx]
                    img_idx = sample['idx'][idx]
                    writer.add_images(
                        tag='Epoch_' + str(epoch) + '/Train_Heatmaps/image_' + str(img_idx) + '_' + groundtruth,
                        img_tensor=orig_viz, dataformats='NHWC', global_step=counter[img_idx])
                    y_scores = nn.Softmax(dim=1)(logits_cl.detach())
                    predicted_categories = y_scores[idx].unsqueeze(0).argmax(dim=1)
                    predicted_cl = [(categories[x], format(y_scores[idx].view(-1)[x], '.4f')) for x in
                                    predicted_categories.view(-1)]
                    labels_cl = [(categories[x], format(y_scores[idx].view(-1)[x], '.4f')) for x in
                                 [(label_idx_list[idx])]]
                    import itertools
                    predicted_cl = list(itertools.chain(*predicted_cl))
                    labels_cl = list(itertools.chain(*labels_cl))
                    cl_text = 'cl_gt_' + '_'.join(labels_cl) + '_pred_' + '_'.join(predicted_cl)
                    am_labels = am_scores.argmax(dim=1)
                    predicted_am = [(categories[x], format(am_scores[idx].view(-1)[x], '.4f')) for x in
                                    am_labels[idx].view(-1)]
                    labels_am = [(categories[x], format(am_scores[idx].view(-1)[x], '.4f')) for x in
                                 [label_idx_list[idx]]]
                    import itertools
                    predicted_am = list(itertools.chain(*predicted_am))
                    labels_am = list(itertools.chain(*labels_am))
                    am_text = '_am_gt_' + '_'.join(labels_am) + '_pred_' + '_'.join(predicted_am)
                    writer.add_text('Train_Heatmaps_Description/image_' + str(img_idx) + '_' + groundtruth,
                                    cl_text + am_text,
                                    global_step=counter[img_idx])
                    counter[img_idx] += 1

        print("pos = {} neg = {}".format(count_pos, count_neg))
        model.train(False)
        j = 0

        test_total_pos_correct = 0
        test_total_neg_correct = 0

        test_differences = np.zeros(len(medt_loader.datasets['test']) * batch_size)
        for sample in medt_loader.datasets['test']:
            label_idx_list = sample['labels']

            batch, _, _ = MedT_preprocess_image_v4(img=sample['images'][0].squeeze().numpy(), train=False, mean=mean,
                                                   std=std)
            for img in sample['images'][1:]:
                input_tensor, input_image, _ = MedT_preprocess_image_v4(img=img.squeeze().numpy(), train=False,
                                                                        mean=mean,
                                                                        std=std)
                batch = torch.cat((batch, input_tensor), dim=0)
            batch = batch.to(device)
            labels = torch.Tensor(label_idx_list).to(device).long()

            logits_cl, logits_am, heatmaps, masks, masked_images = gain(batch, labels)

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
            test_differences[j * batch_size: j * batch_size + len(difference)] = difference

            am_scores = nn.Softmax(dim=1)(logits_am)
            am_labels = am_scores.argmax(dim=1)

            if (j < 50 and j % 2 == 0) or (j > 50 and j % 150 == 0):
                htm = np.uint8(heatmaps[0].squeeze().cpu().detach().numpy() * 255)
                resize = Resize(size=224)
                orig = sample['images'][0].permute([2, 0, 1])
                orig = resize(orig).permute([1, 2, 0])
                np_orig = orig.cpu().detach().numpy()
                visualization, heatmap = show_cam_on_image(np_orig, htm, True)
                viz = torch.from_numpy(visualization).unsqueeze(0)
                orig = orig.unsqueeze(0)
                masked_image = denorm(masked_images[0].detach().squeeze(), mean, std)
                masked_image = (masked_image.squeeze().permute([1, 2, 0]).cpu().detach().numpy() * 255).round().astype(
                    np.uint8)
                masked_image = torch.from_numpy(masked_image).unsqueeze(0)
                orig_viz = torch.cat((orig, viz, masked_image), 0)
                gt = [categories[x] for x in label_idx_list][0]
                writer.add_images(tag='Test_Heatmaps/image_' + str(j) + '_' + gt,
                                  img_tensor=orig_viz, dataformats='NHWC', global_step=epoch)
                y_scores = nn.Softmax(dim=1)(logits_cl.detach())
                predicted_categories = y_scores[0].unsqueeze(0).argmax(dim=1)
                predicted_cl = [(categories[x], format(y_scores[0].view(-1)[x], '.4f')) for x in
                                predicted_categories.view(-1)]
                labels_cl = [(categories[x], format(y_scores[0].view(-1)[x], '.4f')) for x in [(label_idx_list[0])]]
                import itertools
                predicted_cl = list(itertools.chain(*predicted_cl))
                labels_cl = list(itertools.chain(*labels_cl))
                cl_text = 'cl_gt_' + '_'.join(labels_cl) + '_pred_' + '_'.join(predicted_cl)

                predicted_am = [(categories[x], format(am_scores[0].view(-1)[x], '.4f')) for x in am_labels[0].view(-1)]
                labels_am = [(categories[x], format(am_scores[0].view(-1)[x], '.4f')) for x in [label_idx_list[0]]]
                import itertools
                predicted_am = list(itertools.chain(*predicted_am))
                labels_am = list(itertools.chain(*labels_am))
                am_text = '_am_gt_' + '_'.join(labels_am) + '_pred_' + '_'.join(predicted_am)

                writer.add_text('Test_Heatmaps_Description/image_' + str(j) + '_' + gt, cl_text + am_text,
                                global_step=epoch)

            j += 1

        num_test_samples = len(medt_loader.datasets['test']) * batch_size
        print("finished epoch number:")
        print(epoch)

        gain.increase_epoch_count()

        chkpt_path = 'C:/Users/Student1/PycharmProjects/GCAM/checkpoints/batch_GAIN_MedT/'
        pathlib.Path(chkpt_path).mkdir(parents=True, exist_ok=True)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, chkpt_path + 'with_all_am_with_ex')

        # epoch_train_am_ls.append(epoch_train_am_loss / num_train_samples)

        if (test_first_before_train and epoch > 0) or test_first_before_train == False:
            print('Average epoch train am loss: {:.3f}'.format(epoch_train_am_loss / am_count))
            # epoch_train_cl_ls.append(epoch_train_cl_loss / num_train_samples)
            print('Average epoch train cl loss: {:.3f}'.format(epoch_train_cl_loss / (num_train_samples * batch_size)))
            # epoch_train_total_ls.append(epoch_train_total_loss / num_train_samples)
            print('Average epoch train total loss: {:.3f}'.format(epoch_train_total_loss / count_pos))

            # epoch_train_single_accuracy.append(total_train_single_accuracy / num_train_samples)
            print('Average epoch single train accuracy: {:.3f}'.format(total_train_single_accuracy / num_train_samples))

        # epoch_train_multi_accuracy.append(total_train_multi_accuracy / num_train_samples)
        # print('Average epoch multi train accuracy: {:.3f}'.format(total_train_multi_accuracy / num_train_samples))

        # epoch_test_single_accuracy.append(total_test_single_accuracy / num_test_samples)
        print('Average epoch single test accuracy: {:.3f}'.format(total_test_single_accuracy / num_test_samples))

        # epoch_test_multi_accuracy.append(total_test_multi_accuracy / num_test_samples)
        # print('Average epoch multi test accuracy: {:.3f}'.format(total_test_multi_accuracy / num_test_samples))
        pos_count = medt_loader.get_test_pos_count()

        if (test_first_before_train and epoch > 0) or test_first_before_train == False:
            writer.add_scalar('Loss/train/cl_total_loss', epoch_train_cl_loss / (num_train_samples * batch_size), epoch)
            writer.add_scalar('Loss/train/am_total_loss', epoch_train_am_loss / am_count, epoch)
            writer.add_scalar('Accuracy/train/cl_accuracy',
                              total_train_single_accuracy / (num_train_samples * batch_size), epoch)
            writer.add_scalar('Accuracy/train/cl_accuracy_only_pos',
                              train_total_pos_correct / train_total_pos_seen, epoch)
            writer.add_scalar('Accuracy/train/cl_accuracy_only_neg',
                              train_total_neg_correct / train_total_neg_seen, epoch)
            writer.add_scalar('Accuracy/test/cl_accuracy_only_pos',
                              test_total_pos_correct / pos_count, epoch)
            writer.add_scalar('Accuracy/test/cl_accuracy_only_neg',
                              test_total_neg_correct / (num_test_samples-pos_count), epoch)

            all_sens, _ = calc_sensitivity(train_labels, train_differences)
            writer.add_scalar('ROC/train/ROC_0.1', all_sens[0], epoch)
            writer.add_scalar('ROC/train/ROC_0.05', all_sens[1], epoch)
        writer.add_scalar('Accuracy/test/cl_accuracy', total_test_single_accuracy / num_test_samples, epoch)

        ones = torch.ones(pos_count)
        test_labels = torch.zeros(num_test_samples)
        test_labels[0:len(ones)] = ones
        test_labels = test_labels.int()
        all_sens, _ = calc_sensitivity(test_labels.cpu().numpy(), test_differences)
        writer.add_scalar('ROC/test/ROC_0.1', all_sens[0], epoch)
        writer.add_scalar('ROC/test/ROC_0.05', all_sens[1], epoch)

        writer.add_scalar('IOU/train/average_IOU_per_sample', epoch_IOU / (IOU_i - IOU_prev), epoch)


if __name__ == '__main__':
    main()
    print()