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

    target_layer = model.features[-1]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    pl_im = PIL.Image.open('C:/VOC-dataset/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg')
    np_im = np.array(pl_im)
    plt.imshow(np_im)
    plt.show()

    #input_tensor = preprocess_image(np_im, mean=mean, std=std).to(device)
    #input_tensor = torch.from_numpy(np_im).unsqueeze(0).permute([0,3,1,2]).to(device).float()
    #np_im = input_tensor.squeeze().permute(1,2,0).cpu()

    '''
    cam = GradCAM(model=model, target_layer=target_layer, use_cuda=True)
    grayscale_cam = cam(input_tensor=input_tensor, target_category=0)
    visualization, heatmap = show_cam_on_image(np_im, grayscale_cam, True)
    plt.imshow(visualization)
    plt.show()
    plt.imshow(heatmap)
    plt.show()

    target_class = 0

    gcam = Grad_CAM(model=model)
    for i in range(3):
        gcam.forward(input_tensor)
        ids_ = torch.LongTensor([[target_class]] * len(input_tensor)).to(device)
        gcam.backward(ids=ids_)
        regions = gcam.generate(target_layer='features')
        regions_cpu = regions.squeeze().cpu()
        visualization, heatmap = show_cam_on_image(np_im, regions_cpu, True)
        plt.imshow(visualization)
        plt.show()
        plt.imshow(heatmap)
        plt.show()
        print()
    '''

    dataset_path = 'C:/VOC-dataset'
    input_dims = [224, 224]
    batch_size_dict = {'train': 1, 'test': 1}

    rds = data.RawDataset(root_dir=dataset_path,
                          num_workers=0,
                          output_dims=input_dims,
                          batch_size_dict=batch_size_dict)

    epochs = 15


    loss_fn = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    #gcam = Grad_CAM(model=model)

    gcam = GCAM(model=model, grad_layer='features', num_classes=20)

    total_train_loss = []
    total_train_accuracy = []
    total_test_loss = []
    total_test_accuracy = []

    viz_path = 'C:/Users/Student1/PycharmProjects/GCAM/viz'
    pathlib.Path(viz_path).mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        train_accuracy = []
        mean_train_accuracy = []
        test_accuracy = []
        mean_test_accuracy = []
        train_epoch_loss = []
        test_epoch_loss = []
        train_path = 'C:/Users/Student1/PycharmProjects/GCAM/viz/train'
        pathlib.Path(train_path).mkdir(parents=True, exist_ok=True)
        epoch_path = train_path+'/epoch_'+str(epoch)
        pathlib.Path(epoch_path).mkdir(parents=True, exist_ok=True)

        model.train(True)
        i = 0
        for sample in rds.datasets['train']:
            input_tensor = preprocess_image(sample['image'].squeeze().numpy(), mean=mean, std=std).to(device)
            #input_tensor2 = torch.from_numpy(sample['image'].copy()).unsqueeze(0).permute([0, 3, 1, 2]).to(device).float()
            #np_im2 = input_tensor.squeeze().permute(1, 2, 0).cpu()
            label_idx_list = sample['label/idx']
            num_of_labels = len(label_idx_list)

            """
            print(i)
            if i % 100 == 0 :
                img = sample['image']
                plt.imshow(img)
                plt.show()

                for label_idx in label_idx_list:
                    print(categories[label_idx])
    
    
                    gcam.forward(input_tensor)
                    ids_ = torch.LongTensor([[label_idx]] * len(input_tensor)).to(device)
                    gcam.backward(ids=ids_)
                    regions = gcam.generate(target_layer='features')
                    regions_cpu = regions.squeeze().cpu()
                    visualization, heatmap = show_cam_on_image(img, regions_cpu, True)
                    plt.imshow(visualization)
                    plt.show()
                    plt.imshow(heatmap)
                    plt.show()
    
    
                    '''
                    cam = GradCAM(model=model, target_layer=target_layer, use_cuda=True)
                    grayscale_cam = cam(input_tensor=input_tensor, target_category=label_idx)
                    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    plt.imshow(heatmap)
                    plt.show()
                    visualization = show_cam_on_image(img, heatmap)
                    plt.imshow(visualization)
                    plt.show()
                    print()
                    '''
            """

            optimizer.zero_grad()

            labels = torch.Tensor(label_idx_list).to(device).long()

            logits, heatmap = gcam(input_tensor, labels)

            indices = torch.Tensor(label_idx_list).long().to(device)
            class_onehot = torch.nn.functional.one_hot(indices, num_classes).sum(dim=0).unsqueeze(0).float()

            loss = loss_fn(logits, class_onehot)

            loss.backward()

            optimizer.step()

            if i % 100 == 0:
                print(i)
                print('Loss per image: {:.3f}'.format(loss.detach().item()))

                _, y_pred = logits.detach().topk(num_of_labels)
                y_pred = y_pred.view(-1)
                gt, _ = indices.sort(descending=True)
                gt = gt.view(-1)
                acc = (y_pred == gt).sum() / num_of_labels
                train_accuracy.append(acc)

            if i % 200 == 0:
                train_epoch_loss.append(loss.detach().item())
                if len(train_accuracy) > 1:
                    mean_train_accuracy.append(sum(train_accuracy) / len(train_accuracy))
                    print('Average accuracy: {:.3f}'.format(sum(train_accuracy) / len(train_accuracy)))

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

                htm = heatmap.squeeze().cpu().detach().numpy()
                #plt.imshow(htm)
                #plt.show()

                htm = deprocess_image(htm)
                visualization, heatmap = show_cam_on_image(img, htm, True)
                visualization_m = Image.fromarray(visualization)
                visualization_m.save(dir_path+'/'+'vis.jpg')
                #plt.imshow(visualization)
                #plt.show()
                #plt.imshow(heatmap)
                #plt.show()
                #print()


                if len(train_epoch_loss) > 1:
                    #mx = max(epoch_loss)
                    #smooth = [l / mx for l in epoch_loss]
                    plt.plot(train_epoch_loss)
                    plt.savefig(epoch_path+'/epoch_loss.jpg')
                    #plt.plot(smooth)
                    #plt.savefig(epoch_path + '/smooth.jpg')
                    plt.close()
                    plt.plot(mean_train_accuracy)
                    plt.savefig(epoch_path + '/train_accuracy.jpg')
                    plt.close()
            i+=1


        total_train_loss.append(sum(train_epoch_loss)/len(train_epoch_loss))
        plt.plot(total_train_loss)
        plt.savefig(viz_path + '/total_train_loss.jpg')
        total_train_accuracy.append(mean_train_accuracy[-1])
        plt.plot(total_train_loss)
        plt.savefig(viz_path + '/total_train_accuracy.jpg')

        test_path = 'C:/Users/Student1/PycharmProjects/GCAM/viz/test'
        pathlib.Path(test_path).mkdir(parents=True, exist_ok=True)
        epoch_path = test_path + '/epoch_' + str(epoch)
        pathlib.Path(epoch_path).mkdir(parents=True, exist_ok=True)

        model.train(False)
        i = 0
        for sample in rds.datasets['test']:
            input_tensor = preprocess_image(sample['image'].squeeze().numpy(), mean=mean, std=std).to(device)
            # input_tensor2 = torch.from_numpy(sample['image'].copy()).unsqueeze(0).permute([0, 3, 1, 2]).to(device).float()
            # np_im2 = input_tensor.squeeze().permute(1, 2, 0).cpu()
            label_idx_list = sample['label/idx']
            num_of_labels = len(label_idx_list)

            """
            #this code used for initial experiments  GCAM visualization
            print(i)
            if i % 100 == 0 :
                img = sample['image']
                plt.imshow(img)
                plt.show()

                for label_idx in label_idx_list:
                    print(categories[label_idx])


                    gcam.forward(input_tensor)
                    ids_ = torch.LongTensor([[label_idx]] * len(input_tensor)).to(device)
                    gcam.backward(ids=ids_)
                    regions = gcam.generate(target_layer='features')
                    regions_cpu = regions.squeeze().cpu()
                    visualization, heatmap = show_cam_on_image(img, regions_cpu, True)
                    plt.imshow(visualization)
                    plt.show()
                    plt.imshow(heatmap)
                    plt.show()


                    '''
                    cam = GradCAM(model=model, target_layer=target_layer, use_cuda=True)
                    grayscale_cam = cam(input_tensor=input_tensor, target_category=label_idx)
                    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    plt.imshow(heatmap)
                    plt.show()
                    visualization = show_cam_on_image(img, heatmap)
                    plt.imshow(visualization)
                    plt.show()
                    print()
                    '''
            """

            labels = torch.Tensor(label_idx_list).to(device).long()
            logits, heatmap = gcam(input_tensor, labels)

            indices = torch.Tensor(label_idx_list).long().to(device)
            class_onehot = torch.nn.functional.one_hot(indices, num_classes).sum(dim=0).unsqueeze(0).float()

            loss = loss_fn(logits, class_onehot)

            if i % 100 == 0:
                print(i)
                print('Loss per image: {:.3f}'.format(loss.detach().item()))

                _, y_pred = logits.detach().topk(num_of_labels)
                y_pred = y_pred.view(-1)
                gt, _ = indices.sort(descending=True)
                gt = gt.view(-1)
                acc = (y_pred == gt).sum() / num_of_labels
                test_accuracy.append(acc)

            if i % 200 == 0:
                test_epoch_loss.append(loss.detach().item())
                if len(train_accuracy) > 1:
                    mean_test_accuracy.append(sum(test_accuracy) / len(test_accuracy))
                    print('Average accuracy: {:.3f}'.format(sum(test_accuracy) / len(test_accuracy)))

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

                htm = heatmap.squeeze().cpu().detach().numpy()
                # plt.imshow(htm)
                # plt.show()

                htm = deprocess_image(htm)
                visualization, heatmap = show_cam_on_image(img, htm, True)
                visualization_m = Image.fromarray(visualization)
                visualization_m.save(dir_path + '/' + 'vis.jpg')
                # plt.imshow(visualization)
                # plt.show()
                # plt.imshow(heatmap)
                # plt.show()
                # print()

                if len(test_epoch_loss) > 1:
                    # mx = max(epoch_loss)
                    # smooth = [l / mx for l in epoch_loss]
                    plt.plot(test_epoch_loss)
                    plt.savefig(epoch_path + '/epoch_loss.jpg')
                    # plt.plot(smooth)
                    # plt.savefig(epoch_path + '/smooth.jpg')
                    plt.close()
                    plt.plot(mean_test_accuracy)
                    plt.savefig(epoch_path + '/test_accuracy.jpg')
                    plt.close()
            i+=1

        total_test_loss.append(sum(test_epoch_loss) / len(test_epoch_loss))
        plt.plot(total_test_loss)
        plt.savefig(viz_path + '/total_test_loss.jpg')
        total_test_accuracy.append(mean_test_accuracy[-1])
        plt.plot(total_test_loss)
        plt.savefig(viz_path + '/total_test_accuracy.jpg')




if __name__ == '__main__':
    main()
    print()