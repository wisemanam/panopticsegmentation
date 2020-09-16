from torch import cuda

# This file contains the configuration parameters which will be used throughout your experiments
use_cuda = cuda.is_available()

start_iteration = 0
n_iterations = 120000

save_every_n_iters = 2000

batch_size = 12

n_classes = 34

model = 'CapsuleModelNew1'  # 'CapsuleModel'

# model_id = 1 # CapsuleModelNew1 new positional encoding add to final dimensions 
# model_id = 2 # CapsuleModelNew1 new positional encoding concat

model_id = 3 # CapsuleModelNew1, multiply acts*poses
# model_id = 4 # CapsuleModelNew1, baseline batch_size=8
# model_id = 5 # CapsulesModelNewLayers, multiple layers batch_size=12 

# model_id = 6 # CapsuleModelNew1 concatenate batch_size=12
# model_id = 7 # CapsuleModelNew1 add the final two dimensions batch_size=12


save_dir = './SavedModels/Run%d/' % model_id

learning_rate = 1e-3
weight_decay = 0.0

poly_lr_scheduler = True
use_dropout = True
positional_encoding = False

data_dir = './CityscapesData'

num_workers = 8

seg_coef = 1.0
center_coef = 0
regression_coef = 1.0
class_coef = 1.0

h, w = 512, 1024

use_instance = True
