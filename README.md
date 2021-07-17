
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

## Run the script ##

All you need to do to run the algorithm is the following command

```
$ python --batchsize=20 --total_epochs=50 --nepoch=6000 --nepoch_am=100 --nepoch_ex=1 \
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

See the usecase which will be uploaded soon.

## Use case in a Project ##
Link will be uploaded soon.