from torch import cuda

# This file contains the configuration parameters which will be used throughout your experiments
use_cuda = cuda.is_available()

start_iteration = 0
n_iterations = 90000

save_every_n_iters = 1000

batch_size = 12

n_classes = 34

model = 'CapsuleModel4'  # 'CapsuleModel'

# model_id = 1 # CapsuleModel2 2 capsule layers, vote_dim = 128, lmda = 10000
# model_id = 2 # CapsuleModel2 vote_dim = 128, initial capsules = 16
# model_id = 3 # CapsuleModel2 vote_dim = 128
# model_id = 4 # CapsuleModel2 2 capsule layers, with top-down routing
# model_id = 5 # CapsuleModel2, use_top_down_routing = True, no positional encoding
# model_id = 6 # CapsuleModel2 concatenate
# model_id = 7 # CapsuleModel3, upscale before routing (vote_dim = 128)
# model_id = 8 # CapsuleModel2, use_top_down_routing = True, concat, vote_dim=128, lmda=1
model_id = 9 # CapsuelModel4

save_dir = './SavedModels/Run%d/' % model_id

learning_rate = 1e-3
weight_decay = 0.0

poly_lr_scheduler = True
use_dropout = True
positional_encoding = False
positional_encoding_type = 'concat'
n_init_capsules = 8
multiple_capsule_layers = False
vote_dim = 128
lmda = 1
use_top_down_routing = False

data_dir = './CityscapesData'

num_workers = 8

seg_coef = 1.0
center_coef = 0
regression_coef = 1.0
class_coef = 1.0

h, w = 512, 1024

use_instance = True
