from torch import cuda

# This file contains the configuration parameters which will be used throughout your experiments
use_cuda = cuda.is_available()

start_iteration = 0
n_iterations = 120000

save_every_n_iters = 1000

batch_size = 8

n_classes = 34

model = 'CapsuleModel5'  # 'CapsuleModel'

# model_id = 1 # CapsuleModel4, 16 inital capsules, vote_dim_seg and init_capsule_dim = 16, vote_dim=32
# model_id = 2 # CapsuleModel4, top_down_routing=True, vote_dim_seg and init_capsule_dim = 32, vote_dim=128
# model_id = 3 # CapsuleModel4, concat
# model_id = 4 # CapsuleModel4, add final two dimensions
# model_id = 5 # CapsuleModel2, use_top_down_routing = True, no positional encoding
# model_id = 6 # CapsuleModel2 vote_dim = 128, initial capsules = 16

# model_id = 7 # CapsuleModel5, two_stage=False
model_id = 8 # CapsuelModel5, two_stage=n_iter>20k

# model_id = 9 # CapsuleModel4
# model_id = 10 # CapsuleModel4, fgbg_weights are 10 (not 3)

# model_id = 11 # CapsuleModel5, two_stage=True

save_dir = './SavedModels/Run%d/' % model_id

learning_rate = 1e-3
weight_decay = 0.0

poly_lr_scheduler = True
use_dropout = True
positional_encoding = False
positional_encoding_type = 'addition'
n_init_capsules = 8
multiple_capsule_layers = False
vote_dim = 128 #32
lmda = 1
init_capsule_dim = 32 #16
vote_dim_seg = 32 #16
use_top_down_routing = False

data_dir = './CityscapesData'

num_workers = 8

seg_coef = 1.0
regression_coef = 1.0
class_coef = 1.0

h, w = 512, 1024

use_instance = True
