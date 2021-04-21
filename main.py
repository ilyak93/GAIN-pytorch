import PIL.Image
import pathlib
import torch
import torchvision
from torch import nn
from torchvision.models import vgg19, wide_resnet101_2, mobilenet_v2
import numpy as np
import matplotlib.pyplot as plt

from dataloaders import data
from utils.image import show_cam_on_image, preprocess_image, deprocess_image

from models.gcam import GCAM
from PIL import Image
from tensorboardX import SummaryWriter




def main():
    categories = [
                'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
            ]

    num_classes = len(categories)
    device = torch.device('cuda:0')
    model = vgg19(pretrained=True).train().to(device)

    #model = mobilenet_v2(pretrained=True).train().to(device)

    #change the last layer for finetuning
    classifier = model.classifier
    num_ftrs = classifier[-1].in_features
    new_classifier = torch.nn.Sequential(*(list(model.classifier.children())[:-1]), nn.Linear(num_ftrs, num_classes).to(device))
    model.classifier = new_classifier
    model.train()
    # target_layer = model.features[-1]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    pl_im = PIL.Image.open('C:/VOC-dataset/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg')
    np_im = np.array(pl_im)
    plt.imshow(np_im)
    plt.show()

    #input_tensor = preprocess_image(np_im, mean=mean, std=std).to(device)
    #input_tensor = torch.from_numpy(np_im).unsqueeze(0).permute([0,3,1,2]).to(device).float()
    #np_im = input_tensor.squeeze().permute(1,2,0).cpu()


    dataset_path = 'C:/VOC-dataset'
    input_dims = [224, 224]
    batch_size_dict = {'train': 1, 'test': 1}
    rds = data.RawDataset(root_dir=dataset_path,
                          num_workers=0,
                          output_dims=input_dims,
                          batch_size_dict=batch_size_dict)

    num_train_samples = len(rds.datasets['seq_train'])
    print(num_train_samples)

    num_test_samples = len(rds.datasets['seq_test'])
    print(num_test_samples)


    epochs = 15
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    # gcam = GCAM(model=model, grad_layer='features', num_classes=20)

    epoch_train_single_accuracy = []
    epoch_train_multi_accuracy = []
    epoch_test_single_accuracy = []
    epoch_test_multi_accuracy = []


    #viz_path = 'C:/Users/Student1/PycharmProjects/GCAM/exp1'
    #pathlib.Path(viz_path).mkdir(parents=True, exist_ok=True)

    start_writing_iteration = 5

    for epoch in range(epochs):
        total_train_single_accuracy = 0
        total_train_multi_accuracy = 0
        total_test_single_accuracy = 0
        total_test_multi_accuracy = 0


        train_accuracy = []
        mean_train_accuracy = []
        test_accuracy = []
        mean_test_accuracy = []
        train_epoch_loss = []
        test_epoch_loss = []

        train_multi_accuracy = []
        mean_train_multi_accuracy = []
        test_multi_accuracy = []
        mean_test_multi_accuracy = []


        train_path = 'C:/Users/Student1/PycharmProjects/GCAM/exp1/train'
        pathlib.Path(train_path).mkdir(parents=True, exist_ok=True)
        epoch_path = train_path+'/epoch_'+str(epoch)
        pathlib.Path(epoch_path).mkdir(parents=True, exist_ok=True)

        model.train(True)
        i = 0
        for sample in rds.datasets['seq_train']:
            input_tensor = preprocess_image(sample['image'].squeeze().numpy(), mean=mean, std=std).to(device)
            label_idx_list = sample['label/idx']
            num_of_labels = len(label_idx_list)
            optimizer.zero_grad()
            # labels = torch.Tensor(label_idx_list).to(device).long()

            logits = model(input_tensor)
            # logits, heatmap = gcam(input_tensor, labels)
            indices = torch.Tensor(label_idx_list).long().to(device)
            class_onehot = torch.nn.functional.one_hot(indices, num_classes).sum(dim=0).unsqueeze(0).float()

            loss = loss_fn(logits, class_onehot)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(i)
                print('Loss per image: {:.3f}'.format(loss.detach().item()))

                # Single label evaluation
                y_pred = logits.detach().argmax()
                y_pred = y_pred.view(-1)
                gt, _ = indices.sort(descending=True)
                gt = gt.view(-1)
                acc = (y_pred == gt).sum()
                total_train_single_accuracy += acc

                # Multi label evaluation
                _, y_pred_multi = logits.detach().topk(num_of_labels)
                y_pred_multi = y_pred_multi.view(-1)
                acc_multi = (y_pred_multi == gt).sum() / num_of_labels
                total_train_multi_accuracy += acc_multi

                train_accuracy.append(acc)
                train_multi_accuracy.append(acc_multi)


            if i % 200 == 0:
                train_epoch_loss.append(loss.detach().item())
                if len(train_accuracy) > start_writing_iteration:
                    mean_train_accuracy.append(sum(train_accuracy) / len(train_accuracy))
                    mean_train_multi_accuracy.append(sum(train_multi_accuracy) / len(train_multi_accuracy))
                    print('Average train single label accuracy: {:.3f}'.format(sum(train_accuracy) / len(train_accuracy)))
                    print('Average train multi label accuracy: {:.3f}'.format(sum(train_multi_accuracy) / len(train_multi_accuracy)))

                _, y_pred = logits.detach().topk(num_of_labels)
                y_pred = y_pred.view(-1)
                gt, _ = y_pred.sort(descending=True)

                predicted_categories = [categories[x] for x in gt]

                labels = [categories[label_idx] for label_idx in label_idx_list]
                dir_name = str(i)+'_labels_'+'_'.join(labels)+'_predicted_'+'_'.join(predicted_categories) +'_loss_'+str(loss.detach().item())
                dir_path = epoch_path + '/' + dir_name
                pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

                img = sample['image'].squeeze().numpy()
                img_orig = Image.fromarray(img)
                img_orig.save(dir_path + '/' + 'orig.jpg')

                # htm = heatmap.squeeze().cpu().detach().numpy()
                #plt.imshow(htm)
                #plt.show()

                # htm = deprocess_image(htm)
                # visualization, heatmap = show_cam_on_image(img, htm, True)
                # visualization_m = Image.fromarray(visualization)
                # visualization_m.save(dir_path+'/'+'vis.jpg')
                #plt.imshow(visualization)
                #plt.show()
                #plt.imshow(heatmap)
                #plt.show()
                #print()


                if len(train_epoch_loss) > 1:
                    #mx = max(epoch_loss)
                    #smooth = [l / mx for l in epoch_loss]
                    x_loss = np.arange(0, i+1, 200)
                    plt.plot(x_loss, train_epoch_loss)
                    plt.savefig(epoch_path+'/epoch_loss.jpg')
                    #plt.plot(smooth)
                    #plt.savefig(epoch_path + '/smooth.jpg')
                    plt.close()
                    if i % 200 == 0 and i > start_writing_iteration*100:
                        x_acc = np.arange(600, i + 1, 200)
                        plt.plot(x_acc, mean_train_accuracy)
                        plt.savefig(epoch_path + '/train_accuracy.jpg')
                        plt.close()
                        plt.plot(x_acc, mean_train_multi_accuracy)
                        plt.savefig(epoch_path + '/train_multi_accuracy.jpg')
                        plt.close()
            i+=1

        test_path = 'C:/Users/Student1/PycharmProjects/GCAM/exp1/test'
        pathlib.Path(test_path).mkdir(parents=True, exist_ok=True)
        epoch_path = test_path + '/epoch_' + str(epoch)
        pathlib.Path(epoch_path).mkdir(parents=True, exist_ok=True)

        model.train(False)
        i = 0
        for sample in rds.datasets['seq_test']:
            input_tensor = preprocess_image(sample['image'].squeeze().numpy(), mean=mean, std=std).to(device)
            label_idx_list = sample['label/idx']
            num_of_labels = len(label_idx_list)
            optimizer.zero_grad()
            # labels = torch.Tensor(label_idx_list).to(device).long()

            logits = model(input_tensor)
            # logits, heatmap = gcam(input_tensor, labels)
            indices = torch.Tensor(label_idx_list).long().to(device)
            class_onehot = torch.nn.functional.one_hot(indices, num_classes).sum(dim=0).unsqueeze(0).float()

            loss = loss_fn(logits, class_onehot)
            loss.backward()
            optimizer.step()

            if i % 25 == 0:
                print(i)
                print('Loss per image: {:.3f}'.format(loss.detach().item()))

                # Single label evaluation
                y_pred = logits.detach().argmax()
                y_pred = y_pred.view(-1)
                gt, _ = indices.sort(descending=True)
                gt = gt.view(-1)
                acc = (y_pred == gt).sum()
                total_test_single_accuracy += acc

                # Multi label evaluation
                _, y_pred_multi = logits.detach().topk(num_of_labels)
                y_pred_multi = y_pred_multi.view(-1)
                acc_multi = (y_pred_multi == gt).sum() / num_of_labels
                total_test_multi_accuracy += acc_multi

                test_accuracy.append(acc)
                test_multi_accuracy.append(acc_multi)

            if i % 50 == 0:
                test_epoch_loss.append(loss.detach().item())
                if len(test_accuracy) > start_writing_iteration:
                    mean_test_accuracy.append(sum(test_accuracy) / len(test_accuracy))
                    mean_test_multi_accuracy.append(sum(test_multi_accuracy) / len(test_multi_accuracy))
                    print('Average test single label accuracy: {:.3f}'.format(sum(test_accuracy) / len(test_accuracy)))
                    print('Average train multi label accuracy: {:.3f}'.format(sum(test_multi_accuracy) / len(test_multi_accuracy)))

                _, y_pred = logits.detach().topk(num_of_labels)
                y_pred = y_pred.view(-1)
                gt, _ = y_pred.sort(descending=True)

                predicted_categories = [categories[x] for x in gt]

                labels = [categories[label_idx] for label_idx in label_idx_list]
                dir_name = str(i) + '_labels_' + '_'.join(labels) + '_predicted_' + '_'.join(
                    predicted_categories) + '_loss_' + str(loss.detach().item())
                dir_path = epoch_path + '/' + dir_name
                pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

                img = sample['image'].squeeze().numpy()
                img_orig = Image.fromarray(img)
                img_orig.save(dir_path + '/' + 'orig.jpg')

                # htm = heatmap.squeeze().cpu().detach().numpy()
                # plt.imshow(htm)
                # plt.show()

                # htm = deprocess_image(htm)
                # visualization, heatmap = show_cam_on_image(img, htm, True)
                # visualization_m = Image.fromarray(visualization)
                # visualization_m.save(dir_path+'/'+'vis.jpg')
                # plt.imshow(visualization)
                # plt.show()
                # plt.imshow(heatmap)
                # plt.show()
                # print()

                if len(test_epoch_loss) > 1:
                    # mx = max(epoch_loss)
                    # smooth = [l / mx for l in epoch_loss]
                    x_loss = np.arange(0, i + 1, 50)
                    plt.plot(x_loss, test_epoch_loss)
                    plt.savefig(epoch_path + '/epoch_loss.jpg')
                    # plt.plot(smooth)
                    # plt.savefig(epoch_path + '/smooth.jpg')
                    plt.close()
                    if i % 50 == 0 and i > start_writing_iteration * 25:
                        x_acc = np.arange(150, i + 1, 50)
                        plt.plot(x_acc, mean_test_accuracy)
                        plt.savefig(epoch_path + '/test_accuracy.jpg')
                        plt.close()
                        plt.plot(x_acc, mean_test_multi_accuracy)
                        plt.savefig(epoch_path + '/test_multi_accuracy.jpg')
                        plt.close()
            i += 1

        print("finished epoch number:")
        print(epoch)

        epoch_train_single_accuracy.append(total_train_single_accuracy / num_train_samples)
        print('Average epoch single train accuracy: {:.3f}'.format(total_train_single_accuracy / num_train_samples))

        epoch_train_multi_accuracy.append(total_train_multi_accuracy / num_train_samples)
        print('Average epoch multi train accuracy: {:.3f}'.format(total_train_multi_accuracy / num_train_samples))

        epoch_test_single_accuracy.append(total_test_single_accuracy / num_test_samples)
        print('Average epoch single test accuracy: {:.3f}'.format(total_test_single_accuracy / num_test_samples))

        epoch_test_multi_accuracy.append(total_test_multi_accuracy / num_test_samples)
        print('Average epoch multi test accuracy: {:.3f}'.format(total_test_multi_accuracy / num_test_samples))

        plt.plot(epoch_train_single_accuracy)
        plt.savefig(viz_path + '/epoch_train_single_accuracy.jpg')
        plt.close()
        plt.plot(epoch_train_multi_accuracy)
        plt.savefig(viz_path + '/epoch_train_multi_accuracy.jpg')
        plt.close()
        plt.plot(epoch_test_single_accuracy)
        plt.savefig(viz_path + '/epoch_test_single_accuracy.jpg')
        plt.close()
        plt.plot(epoch_test_multi_accuracy)
        plt.savefig(viz_path + '/epoch_test_multi_accuracy.jpg')
        plt.close()




if __name__ == '__main__':
    main()
    print()