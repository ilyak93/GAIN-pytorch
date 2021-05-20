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

from dataloaders import data
from dataloaders.MedTData import MedT_Loader
from metrics.metrics import calc_sensitivity
from utils.image import show_cam_on_image, preprocess_image, deprocess_image, denorm, MedT_preprocess_image

from models.GAIN import GAIN
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

    # model = mobilenet_v2(pretrained=True).train().to(device)

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

    # pl_im = PIL.Image.open('C:/VOC-dataset/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg')
    # np_im = np.array(pl_im)
    # plt.imshow(np_im)
    # plt.show()

    # input_tensor = preprocess_image(np_im, mean=mean, std=std).to(device)
    # input_tensor = torch.from_numpy(np_im).unsqueeze(0).permute([0,3,1,2]).to(device).float()
    # np_im = input_tensor.squeeze().permute(1,2,0).cpu()

    # dataset_path = 'C:/VOC-dataset'
    # input_dims = [224, 224]
    # batch_size_dict = {'train': 1, 'test': 1}
    batch_size = 1
    medt_loader = MedT_Loader('C:/MDT_dataset/SB3_ulcers_mined_roi_mult/', [0.75, 0.25], batch_size=batch_size)

    # num_train_samples = len(rds.datasets['seq_train'])
    # print(num_train_samples)

    # num_test_samples = len(rds.datasets['seq_test'])
    # print(num_test_samples)

    test_first_before_train = True

    epochs = 100
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    gain = GAIN(model=model, grad_layer='features', num_classes=2, pretraining_epochs=10,
                test_first_before_train=test_first_before_train)
    cl_factor = 1
    am_factor = 1
    ex_factor = 1

    # viz_path = 'C:/Users/Student1/PycharmProjects/GCAM/exp2_GAIN_am05_p'
    # pathlib.Path(viz_path).mkdir(parents=True, exist_ok=True)

    # start_writing_iteration = 5

    chkpnt_epoch = 0
    # checkpoint = torch.load('C:/Users/Student1/PycharmProjects/GCAM/checkpoints/4-epoch-chkpnt')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # chkpnt_epoch = checkpoint['epoch']+1

    writer = SummaryWriter(
        "C:/Users/Student1/PycharmProjects/GCAM" + "/MedT_2"
        + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    i = 0
    num_train_samples = 0
    epoch_size = 500
    am_i = 0
    ex_i = 0

    num_of_used_masks = 0
    num_of_masks_to_use = 200

    for epoch in range(chkpnt_epoch, epochs):
        count_pos = 0
        count_neg = 0
        train_differences = np.zeros(epoch_size * batch_size)
        train_labels = np.zeros(epoch_size * batch_size)
        dif_i = 0

        total_train_single_accuracy = 0
        # total_train_multi_accuracy = 0
        total_test_single_accuracy = 0
        # total_test_multi_accuracy = 0

        epoch_train_am_loss = 0
        epoch_train_cl_loss = 0
        epoch_train_total_loss = 0

        model.train(True)
        train_viz = 0
        for sample in medt_loader.datasets['train']:

            if test_first_before_train and epoch == 0:
                i += 1
                break
            elif i != 0 and i % epoch_size == 0:
                i += 1
                break

            label_idx_list = sample[2]
            num_of_labels = len(label_idx_list)

            batch, batch_orig = MedT_preprocess_image(sample[0][0].squeeze().numpy(), train=True, mean=mean, std=std)
            for img in sample[0][1:]:
                input_tensor, input_image = MedT_preprocess_image(img.squeeze().numpy(), train=True, mean=mean, std=std)
                batch = torch.cat((batch, input_tensor), dim=0)
            batch = batch.to(device)

            optimizer.zero_grad()
            labels = torch.Tensor(label_idx_list).to(device).long()

            count_pos += (labels == 1).int().sum()
            count_neg += (labels == 0).int().sum()

            # logits_cl = model(batch)
            logits_cl, logits_am, heatmap, masked_image, mask = gain(batch, labels)

            indices = torch.Tensor(label_idx_list).long().to(device)
            class_onehot = torch.nn.functional.one_hot(indices, num_classes).sum(dim=0).unsqueeze(0).float()

            cl_loss = loss_fn(logits_cl, class_onehot)

            am_scores = nn.Softmax(dim=1)(logits_am)
            _, am_labels = am_scores.argmax(dim=0)
            am_labels_scores = am_scores.view(-1)[labels]
            am_loss = am_labels_scores.sum() / am_labels_scores.size(0)

            # g = make_dot(am_loss, dict(gain.named_parameters()), show_attrs = True, show_saved = True)
            # g.save('grad_viz', train_path)

            total_loss = cl_loss * cl_factor + am_loss * am_factor

            epoch_train_am_loss += (am_loss * am_factor).detach().cpu().item()

            epoch_train_total_loss += total_loss.detach().cpu().item()

            epoch_train_cl_loss += (cl_loss * cl_factor).detach().cpu().item()

            # logits = logits_cl.detach().view(-1)
            difference = (logits_cl[:, 1] - logits_cl[:, 0]).cpu().detach().numpy()
            train_differences[dif_i: dif_i + len(difference)] = difference
            train_labels[dif_i: dif_i + len(difference)] = labels.squeeze().cpu().detach().numpy()
            dif_i += len(difference)

            writer.add_scalar('Loss/train/cl_loss', (cl_loss * cl_factor).detach().cpu().item(), i)
            writer.add_scalar('Loss/train/total_loss', total_loss.detach().cpu().item(), i)

            if gain.AM_enabled() and labels == 1:
                loss = total_loss
                writer.add_scalar('Loss/train/am_loss', (am_loss * am_factor).detach().cpu().item(), am_i)
                am_i += 1
                original_mask = sample[1]
                if original_mask != -1:
                    ex_loss = torch.pow(heatmap - original_mask, 2).sum()
                    writer.add_scalar('Loss/train/ex_loss', (ex_loss * ex_factor).detach().cpu().item(), ex_i)
                    loss += ex_factor * ex_loss
                    ex_i += 1
            else:
                loss = cl_loss

            loss.backward()
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

            if train_viz < 100:
                htm = heatmap.squeeze().cpu().detach().numpy()
                htm = deprocess_image(htm)
                visualization, heatmap = show_cam_on_image(np.asarray(batch_orig), htm, True)
                viz = torch.from_numpy(visualization).unsqueeze(0)
                augmented = torch.tensor(np.asarray(batch_orig)).unsqueeze(0)
                orig = sample[0][0].unsqueeze(0)
                masked_image = denorm(masked_image.detach().squeeze(), mean, std)
                masked_image = (masked_image.squeeze().permute([1, 2, 0]).cpu().detach().numpy() * 255).round().astype(
                    np.uint8)
                masked_image = torch.from_numpy(masked_image).unsqueeze(0)
                if gt == 1:
                    orig_mask = sample[1][0].unsqueeze(2).repeat([1, 1, 3])
                    orig_mask[orig_mask == 255] = 30
                    orig_masked = orig + orig_mask
                    orig_viz = torch.cat((orig, orig_masked, augmented, viz, masked_image), 0)
                else:
                    orig_viz = torch.cat((orig, augmented, viz, masked_image), 0)
                grid = torchvision.utils.make_grid(orig_viz.permute([0, 3, 1, 2]))
                gt = [categories[x] for x in label_idx_list]
                writer.add_image(tag='Train_Heatmaps/image_' + str(i) + '_' + '_'.join(gt),
                                 img_tensor=grid, global_step=epoch,
                                 dataformats='CHW')
                y_scores = nn.Softmax(dim=1)(logits_cl.detach())
                _, predicted_categories = y_scores.argmax(dim=0)
                predicted_cl = [(categories[x], format(y_scores.view(-1)[x], '.4f')) for x in
                                predicted_categories.view(-1)]
                labels_cl = [(categories[x], format(y_scores.view(-1)[x], '.4f')) for x in label_idx_list]
                import itertools
                predicted_cl = list(itertools.chain(*predicted_cl))
                labels_cl = list(itertools.chain(*labels_cl))
                cl_text = 'cl_gt_' + '_'.join(labels_cl) + '_pred_' + '_'.join(predicted_cl)

                predicted_am = [(categories[x], format(am_scores.view(-1)[x], '.4f')) for x in am_labels.view(-1)]
                labels_am = [(categories[x], format(am_scores.view(-1)[x], '.4f')) for x in label_idx_list]
                import itertools
                predicted_am = list(itertools.chain(*predicted_am))
                labels_am = list(itertools.chain(*labels_am))
                am_text = '_am_gt_' + '_'.join(labels_am) + '_pred_' + '_'.join(predicted_am)

                writer.add_text('Train_Heatmaps_Description/image_' + str(i) + '_' + '_'.join(gt), cl_text + am_text,
                                global_step=epoch)
                train_viz += 1

        print("pos = {} neg = {}".format(count_pos, count_neg))
        model.train(False)
        j = 0
        dif_j = 0
        test_differences = np.zeros(len(medt_loader.datasets['test']) * batch_size)
        for sample in medt_loader.datasets['test']:
            label_idx_list = sample[2]
            num_of_labels = len(label_idx_list)

            batch, batch_orig = MedT_preprocess_image(sample[0][0].squeeze().numpy(), train=False, mean=mean, std=std)
            for img in sample[0][1:]:
                input_tensor, input_image = MedT_preprocess_image(img.squeeze().numpy(), train=False, mean=mean,
                                                                  std=std)
                batch = torch.cat((batch, input_tensor), dim=0)
            batch = batch.to(device)
            labels = label_idx_list.to(device).long()

            # logits_cl = model(batch)

            logits_cl, logits_am, heatmap, masked_image, mask = gain(batch, labels)

            # Single label evaluation
            y_pred = logits_cl.detach().argmax(dim=1)
            y_pred = y_pred.view(-1)
            gt = labels.view(-1)
            acc = (y_pred == gt).sum()
            total_test_single_accuracy += acc.detach().cpu()

            difference = (logits_cl[:, 1] - logits_cl[:, 0]).cpu().detach().numpy()
            test_differences[j * batch_size: j * batch_size + len(difference)] = difference

            am_scores = nn.Softmax(dim=1)(logits_am)
            _, am_labels = am_scores.topk(num_of_labels)

            difference = (logits_cl[:, 1] - logits_cl[:, 0]).cpu().detach().numpy()
            train_differences[dif_j: dif_j + len(difference)] = difference
            train_labels[dif_j: dif_j + len(difference)] = labels.squeeze().cpu().detach().numpy()
            dif_j += len(difference)

            # Multi label evaluation
            # _, y_pred_multi = logits_cl.detach().topk(num_of_labels)
            # y_pred_multi = y_pred_multi.view(-1)
            # acc_multi = (y_pred_multi == gt).sum() / num_of_labels
            # total_test_multi_accuracy += acc_multi.detach().cpu()

            '''
            if i % 25 == 0:
                print(i)
                print('Classification Loss per image: {:.3f}'.format(cl_loss.detach().item()))
                print('AM Loss per image: {:.3f}'.format(am_loss.detach().item()))
                print('Total Loss per image: {:.3f}'.format(total_loss.detach().item()))
                test_accuracy.append(acc.detach().cpu())
                #test_multi_accuracy.append(acc_multi.detach().cpu())
            '''

            if (j < 2500 and j % 50 == 0) or (j > 2500 and j % 3000 == 0):
                img = sample[0].squeeze().numpy()
                htm = heatmap.squeeze().cpu().detach().numpy()
                htm = deprocess_image(htm)
                visualization, heatmap = show_cam_on_image(img, htm, True)

                viz = torch.from_numpy(visualization).unsqueeze(0)
                orig = sample[0]
                masked_image = denorm(masked_image.detach().squeeze(), mean, std)
                masked_image = (masked_image.squeeze().permute([1, 2, 0]).cpu().detach().numpy() * 255).round().astype(
                    np.uint8)
                masked_image = torch.from_numpy(masked_image).unsqueeze(0)
                orig_viz = torch.cat((orig, viz, masked_image), 0)
                grid = torchvision.utils.make_grid(orig_viz.permute([0, 3, 1, 2]))
                gt = [categories[x] for x in label_idx_list]
                writer.add_image(tag='Test_Heatmaps/image_' + str(j) + '_' + '_'.join(gt),
                                 img_tensor=grid, global_step=epoch,
                                 dataformats='CHW')
                y_scores = nn.Softmax(dim=1)(logits_cl.detach())
                _, predicted_categories = y_scores.topk(num_of_labels)
                predicted_cl = [(categories[x], format(y_scores.view(-1)[x], '.4f')) for x in
                                predicted_categories.view(-1)]
                labels_cl = [(categories[x], format(y_scores.view(-1)[x], '.4f')) for x in label_idx_list]
                import itertools
                predicted_cl = list(itertools.chain(*predicted_cl))
                labels_cl = list(itertools.chain(*labels_cl))
                cl_text = 'cl_gt_' + '_'.join(labels_cl) + '_pred_' + '_'.join(predicted_cl)

                predicted_am = [(categories[x], format(am_scores.view(-1)[x], '.4f')) for x in am_labels.view(-1)]
                labels_am = [(categories[x], format(am_scores.view(-1)[x], '.4f')) for x in label_idx_list]
                import itertools
                predicted_am = list(itertools.chain(*predicted_am))
                labels_am = list(itertools.chain(*labels_am))
                am_text = '_am_gt_' + '_'.join(labels_am) + '_pred_' + '_'.join(predicted_am)

                writer.add_text('Test_Heatmaps_Description/image_' + str(j) + '_' + '_'.join(gt),
                                cl_text + am_text, global_step=epoch)

            j += 1

        num_test_samples = len(medt_loader.datasets['test']) * batch_size
        print("finished epoch number:")
        print(epoch)

        gain.increase_epoch_count()

        # chkpt_path = 'C:/Users/Student1/PycharmProjects/GCAM/checkpoints/am05_p/'
        # pathlib.Path(chkpt_path).mkdir(parents=True, exist_ok=True)

        # torch.save({
        #    'epoch': epoch,
        #    'model_state_dict': model.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict(),
        # }, chkpt_path + str(epoch))

        # epoch_train_am_ls.append(epoch_train_am_loss / num_train_samples)

        if (test_first_before_train and epoch > 0) or test_first_before_train == False:
            print('Average epoch train am loss: {:.3f}'.format(epoch_train_am_loss / num_train_samples))
            # epoch_train_cl_ls.append(epoch_train_cl_loss / num_train_samples)
            print('Average epoch train cl loss: {:.3f}'.format(epoch_train_cl_loss / num_train_samples))
            # epoch_train_total_ls.append(epoch_train_total_loss / num_train_samples)
            print('Average epoch train total loss: {:.3f}'.format(epoch_train_total_loss / num_train_samples))

            # epoch_train_single_accuracy.append(total_train_single_accuracy / num_train_samples)
            print('Average epoch single train accuracy: {:.3f}'.format(total_train_single_accuracy / num_train_samples))

        # epoch_train_multi_accuracy.append(total_train_multi_accuracy / num_train_samples)
        # print('Average epoch multi train accuracy: {:.3f}'.format(total_train_multi_accuracy / num_train_samples))

        # epoch_test_single_accuracy.append(total_test_single_accuracy / num_test_samples)
        print('Average epoch single test accuracy: {:.3f}'.format(total_test_single_accuracy / num_test_samples))

        # epoch_test_multi_accuracy.append(total_test_multi_accuracy / num_test_samples)
        # print('Average epoch multi test accuracy: {:.3f}'.format(total_test_multi_accuracy / num_test_samples))

        if (test_first_before_train and epoch > 0) or test_first_before_train == False:
            writer.add_scalar('Loss/train/cl_total_loss', epoch_train_cl_loss / (num_train_samples * batch_size), epoch)
            # writer.add_scalar('Loss/train/am_tota_loss', epoch_train_am_loss / num_train_samples, epoch)
            # writer.add_scalar('Loss/train/combined_total_loss', epoch_train_total_loss / num_train_samples, epoch)
            writer.add_scalar('Accuracy/train/cl_accuracy',
                              total_train_single_accuracy / (num_train_samples * batch_size), epoch)

            all_sens, _ = calc_sensitivity(train_labels, train_differences)
            writer.add_scalar('ROC/train/ROC_0.1', all_sens[0], epoch)
            writer.add_scalar('ROC/train/ROC_0.05', all_sens[1], epoch)
        writer.add_scalar('Accuracy/test/cl_accuracy', total_test_single_accuracy / num_test_samples, epoch)

        pos_count = medt_loader.get_test_pos_count()
        ones = torch.ones(pos_count)
        test_labels = torch.zeros(num_test_samples)
        test_labels[0:len(ones)] = ones
        test_labels = test_labels.int()
        all_sens, _ = calc_sensitivity(test_labels.cpu().numpy(), test_differences)
        writer.add_scalar('ROC/test/ROC_0.1', all_sens[0], epoch)
        writer.add_scalar('ROC/test/ROC_0.05', all_sens[1], epoch)


if __name__ == '__main__':
    main()
    print()