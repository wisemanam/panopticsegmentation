import config
import numpy as np
import torch
from train import get_accuracy
from torchvision.utils import save_image
import os
from dataloader import DataLoader, ValidationDataset, get_cityscapes_dataset
from torch.autograd.variable import Variable
from cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling import main as eval_main
from deeplabv3 import DeepLabV3
from PIL import Image

def inference(model, data_loader):
    model.eval()

    if config.use_cuda:
        model.cuda()

    y_list = []
    y_pred_seg_list = []
    img_ind = 0

    for i, sample in enumerate(data_loader):
        image, (y_gt_seg, y_gt_center, y_gt_regression), name = sample
        y_gt_seg = Variable(y_gt_seg.type(torch.LongTensor))
        y_gt_center = Variable(y_gt_center.type(torch.FloatTensor))
        y_gt_regression = Variable(y_gt_regression.type(torch.FloatTensor))
        
        if config.use_cuda:
            image = image.cuda()
            y_gt_seg = y_gt_seg.cuda()
            y_gt_center = y_gt_center.cuda()
            y_gt_regression = y_gt_regression.cuda()
       
        y_pred_seg, y_pred_center, y_pred_regression = model(image)

        y_list.append(name)

        y_pred_seg_softmax = torch.argmax(y_pred_seg, 1)  # should give shape (B, H, W)

        for img in y_pred_seg_softmax:
            img_name = './SavedImages/output_segmentation_%d.png' % img_ind
            save_image(img, img_name)
            img_ind += 1
            y_pred_seg_list.append(img_name)
            
        
    eval_main(y_pred_seg_list, y_list)
    

    print('Finished inference.')

def main():
    max_epoch = 0
    min_loss = 1000000.0000
    for filename in os.listdir('./SavedModels/Run%d/' % config.model_id):
        model_info = filename.split('_')
        epoch = model_info[1]
        loss = model_info[2].split('.p')[0]
        if float(loss) < min_loss:
            min_loss = float(loss)
        if int(epoch) > max_epoch:
            max_epoch = int(epoch)
    model = Model2('Model2', 'SimpleSegmentation/')
    model.load_state_dict(torch.load(os.path.join(config.save_dir, 'model_{}_{:.4f}.pth'.format(max_epoch, min_loss)))['state_dict'])
    val_dataset = get_cityscapes_dataset('~/SimpleSegmentation/CityscapesData', False, download=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    inference(model, val_dataloader)

