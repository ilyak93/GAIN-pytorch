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


    # change the last layer for finetuning
    classifier = model.classifier
    num_ftrs = classifier[-1].in_features
    new_classifier = torch.nn.Sequential(*(list(model.classifier.children())[:-1]),
                                         nn.Linear(num_ftrs, num_classes).to(device))
    model.classifier = new_classifier
    model.train()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    batch_size = 24
    medt_loader = MedT_Loader('C:/MDT_dataset/SB3_ulcers_mined_roi_mult/', [0.75, 0.25], batch_size=batch_size)

    test_first_before_train = True

    epochs = 100
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    gain = GAIN(model=model, grad_layer='features', num_classes=2, pretraining_epochs=10,
                test_first_before_train=test_first_before_train)

    chkpnt_epoch = 0
    # checkpoint = torch.load('C:/Users/Student1/PycharmProjects/GCAM/checkpoints/4-epoch-chkpnt')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # chkpnt_epoch = checkpoint['epoch']+1

    writer = SummaryWriter(
        "C:/Users/Student1/PycharmProjects/GCAM" + "/MedT_pretraining_10_sigma_0.3_omega_10_weighted_debug2" + datetime.datetime.now().strftime(
            '%Y-%m-%d_%H-%M-%S'))
    i=0
    num_train_samples = 0
    epoch_size = 500

    for epoch in range(chkpnt_epoch, epochs):
        count_pos = 0
        count_neg = 0
        train_differences = np.zeros(epoch_size*batch_size)
        train_labels = np.zeros(epoch_size*batch_size)

        total_train_single_accuracy = 0
        total_test_single_accuracy = 0

        epoch_train_cl_loss = 0

        model.train(True)

        for sample in medt_loader.datasets['train']:

            if test_first_before_train and epoch == 0:
                i += 1
                break
            elif i != 0 and i % epoch_size == 0:
                i += 1
                break


            label_idx_list = [sample[2]]

            batch, _ = MedT_preprocess_image(sample[0][0].squeeze().numpy(), train=True, mean=mean, std=std)
            for img in sample[0][1:]:
                input_tensor, input_image = MedT_preprocess_image(img.squeeze().numpy(), train=True, mean=mean, std=std)
                batch = torch.cat((batch,input_tensor), dim=0)
            batch = batch.to(device)
            input_tensor = input_tensor.to(device)
            optimizer.zero_grad()
            labels = torch.Tensor(label_idx_list).to(device).long()

            count_pos += (labels==1).int().sum()
            count_neg += (labels==0).int().sum()

            logits_cl = model(batch)

            lb1 = labels
            lb2 = 1-lb1
            lbs = torch.cat((lb2, lb1), dim=0).transpose(0,1).float()

            cl_loss = loss_fn(logits_cl, lbs)

            epoch_train_cl_loss += cl_loss.detach().cpu().item()


            difference = (logits_cl[:,1] - logits_cl[:,0]).cpu().detach().numpy()
            train_differences[(i % epoch_size) * batch_size : ((i % epoch_size) + 1) * batch_size] = difference
            train_labels[(i % epoch_size) * batch_size : ((i % epoch_size) + 1) * batch_size] = labels.squeeze().cpu().detach().numpy()

            writer.add_scalar('Loss/train/cl_loss', cl_loss.detach().cpu().item(), i)

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

        print("pos = {} neg = {}".format(count_pos, count_neg))
        model.train(False)
        j = 0
        test_differences = np.zeros(len(medt_loader.datasets['test'])*batch_size)
        for sample in medt_loader.datasets['test']:
            label_idx_list = sample[2]

            batch, _ = MedT_preprocess_image(sample[0][0].squeeze().numpy(), train=False, mean=mean, std=std)
            for img in sample[0][1:]:
                input_tensor, input_image = MedT_preprocess_image(img.squeeze().numpy(), train=False, mean=mean, std=std)
                batch = torch.cat((batch, input_tensor), dim=0)
            batch = batch.to(device)
            labels = label_idx_list.to(device).long()

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

        num_test_samples = len(medt_loader.datasets['test'])*batch_size
        print("finished epoch number:")
        print(epoch)


        if (test_first_before_train and epoch > 0) or test_first_before_train == False:
            writer.add_scalar('Loss/train/cl_total_loss', epoch_train_cl_loss / (num_train_samples*batch_size), epoch)
            #writer.add_scalar('Loss/train/am_tota_loss', epoch_train_am_loss / num_train_samples, epoch)
            #writer.add_scalar('Loss/train/combined_total_loss', epoch_train_total_loss / num_train_samples, epoch)
            writer.add_scalar('Accuracy/train/cl_accuracy', total_train_single_accuracy / (num_train_samples*batch_size), epoch)

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