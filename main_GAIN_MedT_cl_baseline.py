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
    MedT_preprocess_image_v3, MedT_preprocess_image_v4

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

    batch_size = 24
    epoch_size = 6000
    medt_loader = MedT_Loader('C:/MDT_dataset/SB3_ulcers_mined_roi_mult/',
                              [0.75, 0.25], batch_size=batch_size,
                              steps_per_epoch=epoch_size)

    test_first_before_train = False


    epochs = 100
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


    cl_factor = 1


    chkpnt_epoch = 0

    #checkpoint = torch.load('C:\Users\Student1\PycharmProjects\GCAM\checkpoints\batch_GAIN_MedT\with_am_no_ex_1_')
    #model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #chkpnt_epoch = checkpoint['epoch']+1
    #gain.cur_epoch = chkpnt_epoch
    #if gain.cur_epoch > gain.am_pretraining_epochs:
    #    gain.enable_am = True
    #if gain.cur_epoch > gain.ex_pretraining_epochs:
    #    gain.enable_ex = True

    writer = SummaryWriter(
        "C:/Users/Student1/PycharmProjects/GCAM" + "/MedT_final_cl_gain_e_6000_b_24_v3_baseline_"
        + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    i = 0
    num_train_samples = 0
    epoch_size = epoch_size*batch_size
    am_i = 0

    for epoch in range(chkpnt_epoch, epochs):
        count_pos = 0
        count_neg = 0
        train_differences = np.zeros(epoch_size * batch_size)
        train_labels = np.zeros(epoch_size * batch_size)
        dif_i = 0


        total_train_single_accuracy = 0
        total_test_single_accuracy = 0



        epoch_train_cl_loss = 0


        model.train(True)

        if not test_first_before_train or (test_first_before_train and epoch != 0):
            for sample in medt_loader.datasets['train']:

                label_idx_list = sample['labels']

                batch, _, _ = \
                    MedT_preprocess_image_v4(
                        img=sample['images'][0].squeeze().permute([2,0,1]),
                        mask=sample['masks'][0].unsqueeze(0),
                        train=True, mean=mean, std=std)


                for img, mask in zip(sample['images'][1:],sample['masks'][1:]):
                    input_tensor, _, _ = \
                        MedT_preprocess_image_v4(img=img.squeeze().permute([2,0,1]),
                                                 mask=mask.unsqueeze(0), train=True,
                                                 mean=mean, std=std)
                    batch = torch.cat((batch, input_tensor), dim=0)



                batch = batch.to(device)

                optimizer.zero_grad()
                labels = torch.Tensor(label_idx_list).to(device).long()

                count_pos += (labels == 1).int().sum()
                count_neg += (labels == 0).int().sum()

                # logits_cl = model(batch)
                lb1 = labels.unsqueeze(0)
                lb2 = 1 - lb1
                lbs = torch.cat((lb2, lb1), dim=0).transpose(0, 1).float()

                logits_cl = model(batch)

                cl_loss = loss_fn(logits_cl, lbs)

                total_loss = cl_loss

                pos_indices = [idx for idx, x in enumerate(sample['labels']) if x == 1]


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

        print("pos = {} neg = {}".format(count_pos, count_neg))
        model.train(False)
        j = 0

        test_differences = np.zeros(len(medt_loader.datasets['test']) * batch_size)
        for sample in medt_loader.datasets['test']:
            label_idx_list = sample['labels']

            batch, _, _ = MedT_preprocess_image_v4(img=sample['images'][0].squeeze().numpy(), train=False, mean=mean, std=std)
            for img in sample['images'][1:]:
                input_tensor, input_image, _ = MedT_preprocess_image_v4(img=img.squeeze().numpy(), train=False, mean=mean,
                                                                  std=std)
                batch = torch.cat((batch, input_tensor), dim=0)
            batch = batch.to(device)
            labels = torch.Tensor(label_idx_list).to(device).long()

            logits_cl = model(batch)

            # Single label evaluation
            y_pred = logits_cl.detach().argmax(dim=1)
            y_pred = y_pred.view(-1)
            gt = labels.view(-1)
            acc = (y_pred == gt).sum()
            total_test_single_accuracy += acc.detach().cpu()

            difference = (logits_cl[:, 1] - logits_cl[:, 0]).cpu().detach().numpy()
            test_differences[j * batch_size: j * batch_size + len(difference)] = difference

            j += 1

        num_test_samples = len(medt_loader.datasets['test']) * batch_size
        print("finished epoch number:")
        print(epoch)


        chkpt_path = 'C:/Users/Student1/PycharmProjects/GCAM/checkpoints/batch_GAIN/'
        pathlib.Path(chkpt_path).mkdir(parents=True, exist_ok=True)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
         }, chkpt_path + 'baseline')

        # epoch_train_am_ls.append(epoch_train_am_loss / num_train_samples)

        if (test_first_before_train and epoch > 0) or test_first_before_train == False:
            print('Average epoch train cl loss: {:.3f}'.format(epoch_train_cl_loss / (num_train_samples*batch_size)))

            print('Average epoch single train accuracy: {:.3f}'.format(total_train_single_accuracy / num_train_samples))

        # epoch_train_multi_accuracy.append(total_train_multi_accuracy / num_train_samples)
        # print('Average epoch multi train accuracy: {:.3f}'.format(total_train_multi_accuracy / num_train_samples))

        # epoch_test_single_accuracy.append(total_test_single_accuracy / num_test_samples)
        print('Average epoch single test accuracy: {:.3f}'.format(total_test_single_accuracy / num_test_samples))

        # epoch_test_multi_accuracy.append(total_test_multi_accuracy / num_test_samples)
        # print('Average epoch multi test accuracy: {:.3f}'.format(total_test_multi_accuracy / num_test_samples))

        if (test_first_before_train and epoch > 0) or test_first_before_train == False:
            writer.add_scalar('Loss/train/cl_total_loss', epoch_train_cl_loss / (num_train_samples*batch_size), epoch)

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