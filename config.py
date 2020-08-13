from torch import cuda

# This file contains the configuration parameters which will be used throughout your experiments
use_cuda = cuda.is_available()

start_iteration = 0
n_iterations = 120000

save_every_n_iters = 1000

batch_size = 8

n_classes = 34

model = 'Model'  # 'CapsuleModel'

model_id = 3
save_dir = './SavedModels/Run%d/' % model_id

learning_rate = 1e-3
weight_decay = 0.0

poly_lr_scheduler = True
use_dropout = True

data_dir = './CityscapesData'

num_workers = 8

seg_coef = 1.0
center_coef = 1.0
regression_coef = 0.01

h, w = 512, 1024

use_instance = True
