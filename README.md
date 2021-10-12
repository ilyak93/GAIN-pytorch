
# Guided-Attention-Inference-Network (GAIN)#

![image](https://user-images.githubusercontent.com/50303550/136835409-81275524-73ea-4e05-bb0c-06611b617eac.png)

## Introduction ##

This is a Pytorch implementation of the GAIN algorithm presented in the following paper ([paper CVPR2018](https://arxiv.org/abs/1802.10171)):

Kunpeng Li, Ziyan Wu, Kuan-Chuan Peng, Jan Ernst, Yun Fu.  
Tell Me Where to Look: Guided Attention Inference Network.

If you use this code in your research, please put a link to this github repository.
```
@article{
	paper implementer    = {Ilya Kotlov},
	title     = {Guided-Attention-Inference-Network},
	journal   = {arXiv:1802.10171},
	year      = {2018},
}
```

In general, you can use the code for any purpose with proper attribution. If you do something interesting with the code, we'll be happy to know. Feel free to contact us.

## Requirement ##

* Python >= 3.7
* [Pytorch](http://pytorch.org/) >= v1.8.1
* [Tensorboard](https://www.tensorflow.org/tensorboard) >= v2.5.0

Ensure on PyTorch & NVidia websites you have the appropriate CUDA drivers and Cudnn version.

## Paper implementation, run instructions, key issues and results ##
In our experiments on VOC 2012 dataset (the dataset mentioned in the paper) we implement only the Attention-Mining part.
For the extra-supervision part, you can see the code used for the Medtronic dataset and get inspired by it,
this part is simplier and will be very similiar (just a pixel-wise sqeared error between the original mask and the returned from the model).
![image](https://user-images.githubusercontent.com/50303550/136822833-933c235c-a6e4-44a5-87ac-12263fb0c218.png)

We implement 5th formula in two ways as the paper doesn't present concrete methodology & implementations details:
![image](https://user-images.githubusercontent.com/50303550/136823043-52df3d1c-e602-4db6-80f1-e6e760bcde87.png)

we also use [AutoAugment](https://arxiv.org/abs/1805.09501) with CIFAR10 policy implemented by [torchvision](https://pytorch.org/vision/stable/transforms.html?highlight=autoaugment#torchvision.transforms.AutoAugment)


The first way can be seen in a more authentic and corresponding approach to the paper (in our opinion):
For each of the labels we compute a separate masked image, thus for example if an image has 3 labels, we compute 3 masked images,
(3 heatmaps, 3 masks and e.t.c) and the loss for that image will be the average of their scores after forwarding them.
This implementation appear and used in the following scripts:
1. [the model](https://github.com/ilyak93/GAIN-pytorch/blob/main/models/batch_GAIN_VOC_mutilabel_singlebatch.py) batch_GAIN_VOC_mutilabel_singlebatch
2. [main run script](https://github.com/ilyak93/GAIN-pytorch/blob/main/main_VOC_multilabel_heatmaps_singlebatch.py) main_VOC_multilabel_heatmaps_singlebatch

It works only with batch size of 1 as in the paper.

The second way which is easier to implement especially for batch size greater then 1 is:
The masked image can be compuited w.r.t all of the labels at once, as much as its score,
then it is devided by the amount of labels as in the first approach. See the technical implementation details
for more understanding how it is done and how both approaches differ.
This approach is implemented in the corresponding scripts:
1. [the model](https://github.com/ilyak93/GAIN-pytorch/blob/main/models/batch_GAIN_VOC.py) batch_GAIN_VOC
2. [main run script](https://github.com/ilyak93/GAIN-pytorch/blob/main/main_GAIN_VOC.py) main_GAIN_VOC

In this approach you can run with batch size even greater then 1.

You can experiment with each one of them.

## Use exmaple with Google Drive & Google Collabratory ##

download the VOC 2012 datset from <a href="https://drive.google.com/drive/folders/1N5vE0AYFcim2TYPNZ6DStYxhVItODwBr">here</a>.<br>
You need approximately 3+ GB of free space. <br>
Put it whereever you like on your Gdrive. <br>
Then use the [GAIN notebook](https://github.com/ilyak93/GAIN-pytorch/blob/main/GAIN_notebook_4collab.ipynb), put it on your collab. <br>
You can fork or clone our repository and then change the path or just use it as is.
<br> Dont forget to restart as it says after installing all the requirements. <br>
Run the main script of two mentioned above with the needed parameters (descreption below). Choose the pathes for tensorboard logging, checkpoints and e.t.c if you like. <br>
We recommend you to synchronize your local storage with Google Drive and then view tensorboard locally, if the tensorboard extension doesnt work from Collab (which can happen). <br>

running the main scripts:
```
%run main_GAIN_VOC.py --dataset_path==/content/drive/MyDrive/VOC-dataset --logging_path=./content/drive/MyDrive/logs --logging_name=my_GAIN_run \
```
--device: choose your device, default gpu is likely to be cuda:0, cpu for cpu and e.t.c <br/>
--batch_size: only for second approach <br/>
--epoch_size: number of iterations per epoch <br/>
--dataset_path: change to your local/Gdrive dataset path <br/>
--logging_path: change to your local/Gdrive tensorboard logging path<br/>
--logging_name: change to your local/Gdrive tensorboard logging directory name<br/>
--checkpoint_file_path_load: change to your local/Gdrive tensorboard checkpoint file full path <br/>
--checkpoint_file_path_save:  change to your local/Gdrive tensorboard checkpoint saving directory full path + name <br/>
--checkpoint_nepoch: each which number of epochs you want to checkpoint <br/>
--workers_num: choose number of threads the dataloader will use (2 is preferable for each computing core) <br/>
--test_first: set 1 if you want to run first the model on the test set before training <br/>
--cl_loss_factor: set classification loss weight (adjustable as in the paper) <br/>
--am_loss_factor: set attention-mining loss weight (adjustable as in the paper) <br/>
--nepoch: total number of epochs to train and test <br/>
--lr: initial learning rate, 0.00001 recommended as a start point for batch size 1, 0.0001 for batch size greater then 1 (for the second approach main script) <br/>
--npretrain: number of epochs to pretrain before using AM <br/>
--record_itr_train: each which number of iterations to log images in training mode <br/>
--record_itr_test: each which number of iterations to log images in test mode <br/>
--nrecord: how much images of a batch to record (second approach only) <br/>
--grads_off: with gradients (as in the paper) or without novel approach of ours (see more details below). <br/>
--grads_magnitude: mode with gradients - set the magnitude of the backpropogated gradients 
  (default 1 as in the paper, otherwise as a hyperparameter to play with) * second approach only <br/>
  
  All the parameters set to default values with which the following results were obtained. For initial run you can change only the parameters of the paths as in the running command.
  

* Gradients on/off issue (see results for reasoning and conclusions):
![image](https://user-images.githubusercontent.com/50303550/136836333-e6756176-0e7a-42f8-a436-a1d0f7233db3.png)
<br>For details see the discussion in the use case. 

  
## Results ##  

The paper presents the following results for baseline and AM training: </br>
![image](https://user-images.githubusercontent.com/50303550/136830047-25e208e5-fd0a-4012-b077-7d8141c4298e.png)

  
For the first approach the results of the baseline training (all epochs without training AM loss, such that it is regular classification with just GradCAM vizualisations),
gradients on training (first 5 training epochs only classification training, 5 and further with AM loss training) and gradients off training (same as previous) the results we've got are:
![image](https://user-images.githubusercontent.com/50303550/136829932-ec599840-0660-486d-a16a-33b64c3c2f7a.png)
</br>
Thus we've got no significant imporvement with using AM-loss, moreover with just classification we've obtained more then 90%, althought
the paper results achieve only 80% in similiar case. 

The heatmaps also show no significant change training with AM-loss and without it when the gradients set on (most likely as it is in the paper,
as the paper doesn't present any other issues on the the gradients backward technique and e.t.c).
There are much more effect on the heatmaps when the gradients are off, although as it can be seen from the graph,
the accuracy droped significantly.

Some examples:

Baseline heatmaps:
![image](https://user-images.githubusercontent.com/50303550/136831639-cffa27f0-a317-4d15-a85e-370942966f89.png)

![image](https://user-images.githubusercontent.com/50303550/136831780-9f6d344b-ad35-4229-bf66-60d9d3c135c9.png)

![image](https://user-images.githubusercontent.com/50303550/136831834-cb7ce515-bbaf-46c9-a98e-4add97bb95f7.png)

AM-training heatmaps with gradients on:
![image](https://user-images.githubusercontent.com/50303550/136832329-db100ba5-6036-4ad7-80b6-aadb44ce2d28.png)

![image](https://user-images.githubusercontent.com/50303550/136832398-185c1f2a-bdcb-434a-80b8-ee8da3b65c1c.png)

![image](https://user-images.githubusercontent.com/50303550/136832484-6e10e4d8-d6cb-4ae7-a252-27bc58b629f9.png)

AM-training heatmaps with gradients off:
![image](https://user-images.githubusercontent.com/50303550/136832650-5c174fec-60ad-4f27-9deb-c7be647cbebc.png)

![image](https://user-images.githubusercontent.com/50303550/136832763-f1bd1d67-3f7a-4bee-9acf-5be9a26d20bb.png)

![image](https://user-images.githubusercontent.com/50303550/136832865-10a1f3f1-0045-48eb-8134-d8ff29e2b7ba.png)

The paper presented the following results: </br>
![image](https://user-images.githubusercontent.com/50303550/136832967-9f28b3d5-80bc-4c43-909b-b0736af5da89.png)

Which is much more similiar to the results with the gradients set off, but the paper also wrote about an improvement
in accuracy, a result which we were not able to reproduce.

The Quantative results of the second approach are:

![image](https://user-images.githubusercontent.com/50303550/136833623-60a55f07-7681-4de4-8bbc-9c64edff5f32.png)

From which we can see pretty much the same behavior as from the first approach.

The heatmaps are pretty much alike.


You can download and look at the full logs from [here](https://drive.google.com/drive/folders/1ZwJmjaUkah_9Q041t1KFYHEFp0z3cMse).


## Run the script on Medtronic endoscopy dataset ##

(to run it with VOC dataset see the last section)

All you need to do to run the algorithm is the following command

```
$ python main_GAIN_MedT.py --batchsize=20 --total_epochs=50 --nepoch=6000 --nepoch_am=100 --nepoch_ex=1 \
         --masks_to_use=1 --lr=0.0001 --pos_to_write_train=50 --neg_to_write_train=20 \
		 --pos_to_write_test=50 --neg_to_write_test=50 --log_name=ex_1_all --cl_weight=1 \
		 --am_weight=1 --ex_weight=1 ----am_on_all=1 --grad_magnitude=1 --test_before_train=0 \
		 --batch_pos_dist=0.25 --input_dir=C:/MDT_dataset/SB3_ulcers_mined_roi_mult/ \ 
		 --output_dir=./ --checkpoint_name=ex_1_all
```
--batchsize: batch size <br/>
--total_epochs: number of total epoch to train <br/>
--nepoch: number of iterations per epoch <br/>
--nepoch_am: number of epochs to train without AM loss <br/>
--nepoch_ex: number of epochs to train without EX loss <br/>
--masks_to_use: relative number of masks to train with EX loss (0-1 of all of the masks) <br/>
--lr: initial learning rate <br/>
--pos/neg_to_write_train/test: how many pos/neg samples to monitor in train-set/test-set <br/>
--pos/neg_to_write_train/test: how many pos/neg samples to monitor in train-set/test-set <br/>
--write_every: how often to write numeric measurements (losses and e.t.c) to tb
--log_name: name of log tb save the monitoring with the current date and time after the name.<br/>
--batch_pos_dist: distribution per batch of positives and by compelement negatives. For example for value 0.25 and 2 labels the distribution per batch is [0.75, 0.25].<br/>
--input_dir: where the data is dislocated with 'traning' and 'validation' folders.<br/>
--output_dir: where the tb output and checkpoints folder will be dislocated.<br/>
 --checkpoint_name: according to 'which output_dir/checkpoints/checkpoint_name+date+time' will be created 
 --cl_weight: weight of classification loss 
--am_weight: weight of attention-mining loss  
--ex_weight: weight of extra-supervision loss 
--am_on_all: train the attention-mining loss on the positives images only or on all the images.
--grad_magnitude: usse to decrease the gradients magnitude of the second path of am loss gadients path
 
 For debug you can put all the files into your IDE in a new project and run this script defining the arguments<br/>
 or run directly the main script will all the arguments embedded and changing them manually and not as arguments to main.<br/>
 The script is `main_GAIN_MedT.py`.

### Visual Examples of monitoring measurements and attention maps visualizations ###

To open tensorboard monitoring, open `Anaconda Prompt` or any other shell like cmd and e.t.c<br/>
Run the following command:
```
tensorboard --port=6009  --samples_per_plugin images=999999,text=999999 --logdir={log_folder_name}
```

For example:
```
tensorboard --port=6009  --samples_per_plugin images=999999,text=999999 \
 --logdir=C:\Users\Student1\PycharmProjects\GCAM\MedT_final_cl_gain_e_6000_b_24_v3_with_all_am_with_ex_pretrain_2_1_sigma_0.6_omega_30_grad_2021-07-01_22-48-56`
```

where instead of '{log_folder_name}', put your own logging which you named using this --log_name argument.<br/>
use the full path if you have troubles with relative.

Choose any port you would like, if you erase this parameter it will choose the free port for you.
--samples_per_plugin images=999999,text=999999 parameter ensures that you can view in your tensorboard <br/>as much images and text steps as you want.

A few examples of current monitoring format.

Loss & ROC & IOU & Accuracy monitoring (pretty much as expected):<br>
<a href="https://ibb.co/z47rPfR"><img src="https://i.ibb.co/DR8twKC/Num.jpg" alt="Num" border="0"></a>

Attention Maps visualizations:<br>
<a href="https://ibb.co/z47rPfR"><img src="https://i.ibb.co/DR8twKC/Num.jpg" alt="Num" border="0"></a>

<a href="https://ibb.co/ySJp0mm"><img src="https://i.ibb.co/ZmCSdRR/viz3.jpg" alt="viz3" border="0"></a>
<a href="https://ibb.co/dmqmNWb"><img src="https://i.ibb.co/Sf8fksQ/viz1.jpg" alt="viz1" border="0"></a>
<a href="https://ibb.co/591jKpm"><img src="https://i.ibb.co/xs2gF9W/viz2.jpg" alt="viz2" border="0"></a>

Attention Maps visualizations descriptions & additional info:
<a href="https://ibb.co/WFSZqK2"><img src="https://i.ibb.co/pZCcGXL/viz5.jpg" alt="viz5" border="0"></a>
<a href="https://ibb.co/4gfYJCL"><img src="https://i.ibb.co/vvBQDfW/viz4.jpg" alt="viz4" border="0"></a>


### For more: ###

See the use-case.

## Use case in an Industrial & Academic Project ##
Link : <a href="https://www.cs.technion.ac.il/~cs234313/projects_sites/S21/07/site/">Using GAIN to improve endoscopy classification results with localization</a>

 [@Ilya Kotlov](https://www.linkedin.com/in/ilyak93/)
