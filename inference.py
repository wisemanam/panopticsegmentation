import config
import numpy as np
import torch
from train import get_accuracy
from torch.autograd.variable import Variable
from torchvision.utils import save_image
import os
from dataloader import DataLoader, ValidationDataset, get_cityscapes_dataset
from createInstanceMaps import create_instance_maps
from cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling import main as eval_main
from cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling import main as eval_main2
from deeplabv3 import DeepLabV3, Model2
from PIL import Image

def inference(model, data_loader):
    model.eval()

    if config.use_cuda:
        model.cuda()

    y_list = []
    y_pred_seg_list = []
    y_pred_instance_list = []
    img_ind = 0

    for i, sample in enumerate(data_loader):
        image, (y_gt_seg, y_gt_center, y_gt_regression, y_gt_reg_pres), name = sample

        if config.use_cuda:
            image = image.cuda()
            y_gt_seg = y_gt_seg.cuda()
            y_gt_center = y_gt_center.cuda()
            y_gt_regression = y_gt_regression.cuda()
            y_gt_reg_pres = y_gt_reg_pres.cuda()

        y_pred_seg, y_pred_center, y_pred_regression = model(image)

        # y_list.append(name)

        y_pred_seg_argmax = torch.argmax(y_pred_seg, 1)  # should give shape (B, H, W)
        y_pred_center = y_pred_center.data.cpu().numpy() # converts tensor to numpy
        y_pred_regression = y_pred_regression.data.cpu().numpy() #converts tensor to numpy
        for j in range(len(y_pred_seg_argmax)):
            img_name = './SavedImages/output_segmentation_%d.png' % img_ind
            img = y_pred_seg_argmax[j].cpu().data.numpy()  # Converts the tensor to numpy
            img  = Image.fromarray(img) # Converts numpy array to PIL Image
            img = img.resize(size=(512, 1024), resample=Image.NEAREST) # Resizes image
            img.save(img_name) # Saves image
            y_pred_seg_list.append(img_name)
            pred_instance_map = create_instance_maps(y_pred_seg_argmax[j], y_pred_center[j], y_pred_regression[j])
            img_name = './SavedImages/output_instances_%d.png' % img_ind
            img  = Image.fromarray(pred_instance_map) # Converts numpy array to PIL Image
            img = img.resize(size=(512, 1024), resample=Image.NEAREST) # Resizes image
            img.save(img_name) # Saves image
            img_ind += 1
            y_pred_instance_list.append(img_name)
            y_list.append(name[j])
        if i == 5:
            break
    eval_main(y_pred_seg_list, y_list)
    eval_main2(y_pred_instance_list, y_list)

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


