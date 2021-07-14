
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

## Run the script ##

All you need to do to run the algorithm is the following command

```
$ python main_MedT.py --batchsize=24 --total_epochs=50 --nepoch=6000 --nepoch_am=100 --nepoch_ex=1 \
			--masks_to_use=0.1 --lr=0.0001 --pos_to_write_train=50 --neg_to_write_train=20 \
			--pos_to_write_test=50 --neg_to_write_test=50 --log_name=my_tensorboard_log --test_before_train=0 \
			--batch_pos_dist=0.25 --input_dir=C:/MDT_dataset/SB3_ulcers_mined_roi_mult/ \
			--output_dir=./ --checkpoint_name=my_checkpoint
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
--log_name: name of log tb save the monitoring with the current date and time after the name.<br/>
--batch_pos_dist: distribution per batch of positives and by compelement negatives. For example for value 0.25 and 2 labels the distribution per batch is [0.75, 0.25].<br/>
--input_dir: where the data is dislocated with 'traning' and 'validation' folders.<br/>
--output_dir: where the tb output and checkpoints folder will be dislocated.<br/>
 --checkpoint_name: according to 'which output_dir/checkpoints/checkpoint_name+date+time' will be created 

### Visual Examples of monitoring measurements and attention maps visualizations ###

A few examples of current monitoring format.

Loss % ROC % IOU % Accuracy monitoring (pretty much as expected):


![Numeric measuremnets monitoring](https://ibb.co/z47rPfR)


### Pre ###

See `easle.py` for the previou


Wipochs like:
![shift with autoencoder](https://i.imgur.com/M1.gif)


## Use case in a Project ##
Link will be uploaded soon.