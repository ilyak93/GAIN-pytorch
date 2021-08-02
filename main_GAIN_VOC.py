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
from utils.image import show_cam_on_image, preprocess_image, deprocess_image, denorm

from models.batch_GAIN_VOC import batch_GAIN_VOC
from PIL import Image

from torch.utils.tensorboard import SummaryWriter
import datetime


def main():
    categories = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
        'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    num_classes = len(categories)
    device = torch.device('cuda:0')
    model = vgg19(pretrained=True).train().to(device)


    # change the last layer for finetuning
    classifier = model.classifier
    num_ftrs = classifier[-1].in_features
    new_classifier = torch.nn.Sequential(*(list(model.classifier.children())[:-1]),
                                         nn.Linear(num_ftrs, num_classes).to(device))
    model.classifier = new_classifier
    model.train()


    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    batch_size = 8
    epoch_size = 500
    dataset_path = 'C:/VOC-dataset'
    input_dims = [224, 224]
    batch_size_dict = {'train': batch_size, 'test': batch_size}
    rds = data.RawDataset(root_dir=dataset_path,
                          num_workers=0,
                          output_dims=input_dims,
                          batch_size_dict=batch_size_dict)

    test_first_before_train = False

    cl_factor = 1
    am_factor = 1

    epochs = 100
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    gain = batch_GAIN_VOC(model=model, grad_layer='features', num_classes=20, pretraining_epochs=5,
                test_first_before_train=test_first_before_train)

    chkpnt_epoch = 0
    # checkpoint = torch.load('C:/Users/Student1/PycharmProjects/GCAM/checkpoints/4-epoch-chkpnt')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # chkpnt_epoch = checkpoint['epoch']+1

    writer = SummaryWriter(
        "C:/Users/Student1/PycharmProjects/GCAM" + "/VOC_multibatch_GAIN_without_resize" +
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    i=0
    num_train_samples = 0



    for epoch in range(chkpnt_epoch, epochs):

        train_viz = 0

        total_train_single_accuracy = 0
        total_test_single_accuracy = 0

        epoch_train_cl_loss = 0

        model.train(True)


        if not test_first_before_train or (test_first_before_train and epoch != 0):

            total_train_single_accuracy = 0
            total_train_multi_accuracy = 0
            total_test_single_accuracy = 0
            total_test_multi_accuracy = 0

            epoch_train_am_loss = 0
            epoch_train_cl_loss = 0
            epoch_train_total_loss = 0

            for sample in rds.datasets['rnd_train']:
                augmented_batch = []
                batch, augmented = preprocess_image(sample[0][0].squeeze().cpu().detach().numpy(), train=True, mean=mean, std=std)
                augmented_batch.append(augmented)
                for img in sample[0][1:]:
                    input_tensor, augmented_image = preprocess_image(img.squeeze().cpu().detach().numpy(), train=True, mean=mean, std=std)
                    batch = torch.cat((batch,input_tensor), dim=0)
                    augmented_batch.append(augmented_image)
                batch = batch.to(device)
                #input_tensor = input_tensor.to(device)
                optimizer.zero_grad()

                labels = sample[2]


                logits_cl, logits_am, heatmap, masked_image, mask = gain(batch, sample[1])

                class_onehot = torch.stack(sample[1]).float()

                cl_loss = loss_fn(logits_cl, class_onehot)

                am_scores = nn.Softmax(dim=1)(logits_am)
                batch_am_lables = []
                batch_am_labels_scores = []
                for k in range(len(batch)):
                    num_of_labels = len(sample[2][k])
                    _, am_labels = am_scores[k].topk(num_of_labels)
                    batch_am_lables.append(am_labels)
                    am_labels_scores = am_scores[k].view(-1)[labels[k]].sum() / num_of_labels
                    batch_am_labels_scores.append(am_labels_scores)
                am_loss = sum(batch_am_labels_scores) / batch_size

                # g = make_dot(am_loss, dict(gain.named_parameters()), show_attrs = True, show_saved = True)
                # g.save('grad_viz', train_path)

                total_loss = cl_loss * cl_factor + am_loss * am_factor

                epoch_train_am_loss += (am_loss * am_factor).detach().cpu().item()
                epoch_train_cl_loss += (cl_loss * cl_factor).detach().cpu().item()
                epoch_train_total_loss += total_loss.detach().cpu().item()

                writer.add_scalar('Per_Step/train/cl_loss', (cl_loss * cl_factor).detach().cpu().item(), i)
                writer.add_scalar('Per_Step/train/am_loss', (am_loss * am_factor).detach().cpu().item(), i)
                writer.add_scalar('Per_Step/train/total_loss', total_loss.detach().cpu().item(), i)

                loss = cl_loss
                loss.backward()
                optimizer.step()

                # Single label evaluation
                for k in range(len(batch)):
                    num_of_labels = len(sample[2][k])
                    _, y_pred = logits_cl[k].detach().topk(k=num_of_labels)
                    y_pred = y_pred.view(-1)
                    gt = torch.tensor(sorted(sample[2][k]), device=device)

                    acc = (y_pred == gt).sum()
                    total_train_single_accuracy += acc.detach().cpu()


                # Multi label evaluation
                #_, y_pred_multi = logits_cl.detach().topk(num_of_labels)
                #y_pred_multi = y_pred_multi.view(-1)
                #acc_multi = (y_pred_multi == gt).sum() / num_of_labels
                #total_train_multi_accuracy += acc_multi.detach().cpu()

                if train_viz < 2:
                    for t in range(len(batch)):
                        num_of_labels = len(sample[2][t])
                        one_heatmap = heatmap[t].squeeze().cpu().detach().numpy()

                        one_augmented_im = torch.tensor(np.array(augmented_batch[t])).to(device).unsqueeze(0)
                        one_masked_image = masked_image[t].detach().squeeze()
                        htm = deprocess_image(one_heatmap)
                        visualization, red_htm = show_cam_on_image(one_augmented_im.cpu().detach().numpy(), htm, True)
                        #plt.imshow(red_htm)
                        #plt.show()
                        #plt.close()
                        #plt.imshow(visualization[0])
                        #plt.show()
                        #plt.close()
                        viz = torch.from_numpy(visualization).to(device)
                        masked_im = denorm(one_masked_image, mean, std)
                        masked_im = (masked_im.squeeze().permute([1, 2, 0])
                            .cpu().detach().numpy() * 255).round()\
                            .astype(np.uint8)
                        #plt.imshow(masked_im)
                        #plt.show()
                        #plt.close()
                        orig = sample[0][t].unsqueeze(0)
                        masked_im = torch.from_numpy(masked_im).unsqueeze(0).to(device)
                        orig_viz = torch.cat((orig, one_augmented_im, viz, masked_im), 0)
                        grid = torchvision.utils.make_grid(orig_viz.permute([0, 3, 1, 2]))
                        gt = [categories[x] for x in labels[t]]
                        writer.add_image(tag='Train_Heatmaps/image_' + str(i) +
                                             '_' + str(t) + '_'  +'_'.join(gt),
                                         img_tensor=grid, global_step=epoch,
                                         dataformats='CHW')
                        y_scores = nn.Softmax()(logits_cl[t].detach())
                        _, predicted_categories = y_scores.topk(num_of_labels)
                        predicted_cl = [(categories[x], format(y_scores.view(-1)[x], '.4f')) for x in
                                        predicted_categories.view(-1)]
                        labels_cl = [(categories[x], format(y_scores.view(-1)[x], '.4f')) for x in labels[t]]
                        import itertools
                        predicted_cl = list(itertools.chain(*predicted_cl))
                        labels_cl = list(itertools.chain(*labels_cl))
                        cl_text = 'cl_gt_' + '_'.join(labels_cl) + '_pred_' + '_'.join(predicted_cl)

                        predicted_am = [(categories[x], format(am_scores[t].view(-1)[x], '.4f')) for x in
                                        batch_am_lables[t].view(-1)]
                        labels_am = [(categories[x], format(am_scores.view(-1)[x], '.4f')) for x in labels[t]]
                        import itertools
                        predicted_am = list(itertools.chain(*predicted_am))
                        labels_am = list(itertools.chain(*labels_am))
                        am_text = '_am_gt_' + '_'.join(labels_am) + '_pred_' + '_'.join(predicted_am)

                        writer.add_text('Train_Heatmaps_Description/image_' + str(i) + '_' + str(t) + '_' +
                                        '_'.join(gt), cl_text + am_text, global_step=epoch)
                    train_viz += 1
                i += 1

                if epoch == 0 and test_first_before_train == False:
                    num_train_samples += 1
                if epoch == 1 and test_first_before_train == True:
                    num_train_samples += 1

                if i % epoch_size == 0:
                    break


        model.train(False)
        j = 0

        for sample in rds.datasets['seq_test']:

            batch, _ = preprocess_image(sample[0][0].squeeze().cpu().detach().numpy(), train=False, mean=mean, std=std)
            for img in sample[0][1:]:
                input_tensor, input_image = preprocess_image(img.squeeze().cpu().detach().numpy(), train=False,
                                                             mean=mean, std=std)
                batch = torch.cat((batch, input_tensor), dim=0)
            batch = batch.to(device)
            labels = sample[2]

            logits_cl, logits_am, heatmap, masked_image, mask = gain(batch, sample[1])

            am_scores = nn.Softmax(dim=1)(logits_am)
            batch_am_lables = []
            for k in range(len(batch)):
                num_of_labels = len(sample[2][k])
                _, am_labels = am_scores[k].topk(num_of_labels)
                batch_am_lables.append(am_labels)


            # Single label evaluation
            for k in range(len(batch)):
                num_of_labels = len(sample[2][k])
                _, y_pred = logits_cl[k].detach().topk(k=num_of_labels)
                y_pred = y_pred.view(-1)
                gt = torch.tensor(sorted(sample[2][k]), device=device)

                acc = (y_pred == gt).sum()
                total_test_single_accuracy += acc.detach().cpu()

            if j % 10 == 0:
                num_of_labels = len(sample[2][0])
                one_heatmap = heatmap[0].squeeze().cpu().detach().numpy()
                one_input_image = sample[0][0].cpu().detach().numpy()
                one_masked_image = masked_image[0].detach().squeeze()
                htm = deprocess_image(one_heatmap)
                visualization, heatmap = show_cam_on_image(one_input_image, htm, True)
                viz = torch.from_numpy(visualization).unsqueeze(0).to(device)
                augmented = torch.tensor(one_input_image).unsqueeze(0).to(device)
                masked_image = denorm(one_masked_image, mean, std)
                masked_image = (
                        masked_image.squeeze().permute([1, 2, 0]).cpu().detach().numpy() * 255).round().astype(
                    np.uint8)
                orig = sample[0][0].unsqueeze(0)
                masked_image = torch.from_numpy(masked_image).unsqueeze(0).to(device)
                orig_viz = torch.cat((orig, augmented, viz, masked_image), 0)
                grid = torchvision.utils.make_grid(orig_viz.permute([0, 3, 1, 2]))
                gt = [categories[x] for x in labels[0]]
                writer.add_image(tag='Test_Heatmaps/image_' + str(j) + '_' + '_'.join(gt),
                                 img_tensor=grid, global_step=epoch,
                                 dataformats='CHW')
                y_scores = nn.Softmax()(logits_cl[0].detach())
                _, predicted_categories = y_scores.topk(num_of_labels)
                predicted_cl = [(categories[x], format(y_scores.view(-1)[x], '.4f')) for x in
                                predicted_categories.view(-1)]
                labels_cl = [(categories[x], format(y_scores.view(-1)[x], '.4f')) for x in labels[0]]
                import itertools
                predicted_cl = list(itertools.chain(*predicted_cl))
                labels_cl = list(itertools.chain(*labels_cl))
                cl_text = 'cl_gt_' + '_'.join(labels_cl) + '_pred_' + '_'.join(predicted_cl)

                predicted_am = [(categories[x], format(am_scores[0].view(-1)[x], '.4f')) for x in
                                batch_am_lables[0].view(-1)]
                labels_am = [(categories[x], format(am_scores.view(-1)[x], '.4f')) for x in labels[0]]
                import itertools
                predicted_am = list(itertools.chain(*predicted_am))
                labels_am = list(itertools.chain(*labels_am))
                am_text = '_am_gt_' + '_'.join(labels_am) + '_pred_' + '_'.join(predicted_am)

                writer.add_text('Test_Heatmaps_Description/image_' + str(j) + '_' + '_'.join(gt),
                                cl_text + am_text, global_step=epoch)



            j += 1

        num_test_samples = len(rds.datasets['seq_test'])*batch_size
        print("finished epoch number:")
        print(epoch)


        if (test_first_before_train and epoch > 0) or test_first_before_train == False:
            writer.add_scalar('Loss/train/cl_total_loss', epoch_train_cl_loss / (num_train_samples*batch_size), epoch)
            writer.add_scalar('Loss/train/am_tota_loss', epoch_train_am_loss / num_train_samples, epoch)
            writer.add_scalar('Loss/train/combined_total_loss', epoch_train_total_loss / num_train_samples, epoch)
            writer.add_scalar('Accuracy/train/cl_accuracy', total_train_single_accuracy / (num_train_samples*batch_size), epoch)


        writer.add_scalar('Accuracy/test/cl_accuracy', total_test_single_accuracy / num_test_samples, epoch)




if __name__ == '__main__':
    main()
    print()