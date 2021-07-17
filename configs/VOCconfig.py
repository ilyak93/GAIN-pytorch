from easydict import EasyDict as edict

__C = edict()

cfg = __C

#######
# OPTIONS FROM RCC CODE
#######
__C.RCC = edict()

#######
# MISC OPTIONS
#######

# For reproducibility
__C.RNG_SEED = 50

# Root directory
#__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__),'..'))





# Number of pairs per batch
__C.BATCHSIZE = 24


__C.INPUTDIR = './'
__C.OUTPUTDIR = './results/'

