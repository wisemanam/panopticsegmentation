from torch import cuda

# This file contains the configuration parameters which will be used throughout your experiments
use_cuda = cuda.is_available()

start_epoch = 1
n_epochs = 50
batch_size = 16

model_id = 1
save_dir = './SavedModels/Run%d/' % model_id

learning_rate = 1e-3
weight_decay = 1e-7

data_dir = '../../Datasets/CityScapes'

num_workers = 8
