
# Guided-Attention-Inference-Network (GAIN)#

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

Ensure on PyTorch & NVdidia websites you have the appropriate CUDA drivers and Cudnn version.

## Paper implementation, run instructions, key issues and results ##
In our experiments on VOC 2012 dataset (the dataset mentioned in the paper) we implement only the Attention-Mining part.
![image](https://user-images.githubusercontent.com/50303550/136822833-933c235c-a6e4-44a5-87ac-12263fb0c218.png)

We implement 5th formula in two ways as the paper doesn't present concrete methodology & implementations details:
![image](https://user-images.githubusercontent.com/50303550/136823043-52df3d1c-e602-4db6-80f1-e6e760bcde87.png)

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

download the VOC 2012 datset from <a href="https://drive.google.com/drive/folders/1N5vE0AYFcim2TYPNZ6DStYxhVItODwBr">here</a>.
You need approximately 3+ GB of free space. 
Put it whereever you like on your Gdrive.
Then use the [GAIN notebook](https://github.com/ilyak93/GAIN-pytorch/blob/main/GAIN_notebook_4collab.ipynb), put it on your collab.
You can fork or clone our repository and then change the path or just use it as is. Dont forget to restart as it says after installing all the requirements.
Run the main script of two mentioned above with the needed parameters (descreption below). Choose the pathes for tensorboard logging, checkpoints and e.t.c if you like.
We recommend you to synchronize your local storage with Google Drive and then view tensorboard locally, if the3 extension doesnt work from Collab (which can happen).

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
  (default 1 as in the paper, otherwise as a hyperparameter to play with) * second approach only



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

Loss & ROC & IOU & Accuracy monitoring (pretty much as expected):
<a href="https://ibb.co/z47rPfR"><img src="https://i.ibb.co/DR8twKC/Num.jpg" alt="Num" border="0"></a>

Attention Maps visualizations:
<a href="https://ibb.co/z47rPfR"><img src="https://i.ibb.co/DR8twKC/Num.jpg" alt="Num" border="0"></a>

<a href="https://ibb.co/ySJp0mm"><img src="https://i.ibb.co/ZmCSdRR/viz3.jpg" alt="viz3" border="0"></a>
<a href="https://ibb.co/dmqmNWb"><img src="https://i.ibb.co/Sf8fksQ/viz1.jpg" alt="viz1" border="0"></a>
<a href="https://ibb.co/591jKpm"><img src="https://i.ibb.co/xs2gF9W/viz2.jpg" alt="viz2" border="0"></a>

Attention Maps visualizations descreptions & additional info:
<a href="https://ibb.co/WFSZqK2"><img src="https://i.ibb.co/pZCcGXL/viz5.jpg" alt="viz5" border="0"></a>
<a href="https://ibb.co/4gfYJCL"><img src="https://i.ibb.co/vvBQDfW/viz4.jpg" alt="viz4" border="0"></a>


### For more: ###

See the use-case.

## Use case in an Industrial & Academic Project ##
Link : <a href="https://www.cs.technion.ac.il/~cs234313/projects_sites/S21/07/site/">Using GAIN to improve endoscopy classification results with localization</a>
