from torch import cuda

# This file contains the configuration parameters which will be used throughout your experiments
use_cuda = cuda.is_available()

start_iteration = 0
n_iterations = 120000

save_every_n_iters = 1000

batch_size = 12

n_classes = 91

model = 'CapsuleModel4'

# model_id = 1 # CapsuleModel6, two_stage=False
# model_id = 2 # CapsuleModel6, two_stage, gradient stopped
# model_id = 3 # NewCapsuleModel6, two_stage, stop_grad = True
# model_id = 4 # NewCapsuleModel6, stop_grad = False
# model_id = 5 # CapsuleModel7
model_id = 6 # CapsuleModel4 with COCO

save_dir = './SavedModels/Run%d/' % model_id

learning_rate = 1e-3
weight_decay = 0.0

poly_lr_scheduler = True
use_dropout = True
positional_encoding = False
positional_encoding_type = 'addition'
n_init_capsules = 8
multiple_capsule_layers = False
vote_dim = 128
lmda = 1
init_capsule_dim = 32
vote_dim_seg = 32
use_top_down_routing = False
stop_grad = True

data_dir = './CocoData'

num_workers = 8

seg_coef = 1.0
regression_coef = 1.0
class_coef = 1.0

h, w = 640, 640

use_instance = True
