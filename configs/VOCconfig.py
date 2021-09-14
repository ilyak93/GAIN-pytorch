from easydict import EasyDict as edict

__C = edict()

cfg = __C




#######
# MISC OPTIONS
#######

# For reproducibility

# Root directory
#__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__),'..'))





# Number of pairs per batch
__C.BATCHSIZE = 24


__C.INPUTDIR = './'
__C.OUTPUTDIR = './results/'

__C.CATEGORIES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
        'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

__C.MEAN = [0.485, 0.456, 0.406]
__C.STD = [0.229, 0.224, 0.225]

__C.INPUT_DIMS = [224, 224]

