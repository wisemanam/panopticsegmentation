import config
import numpy as np
import torch
from train import get_accuracy
from torchvision.utils import save_image
import os
from dataloader import DataLoader, ValidationDataset, get_cityscapes_dataset
from createInstanceMaps import get_centers, create_instance_maps
from cityscapesScripts.cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling import main as eval_main
from cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling import main as eval_main2
from deeplabv3 import DeepLabV3, Model2

def inference(model, data_loader):
    model.eval()

    if config.use_cuda:
        model.cuda()

    y_list = []
    y_pred_seg_list = []
    y_pred_instance_list = []
    y_gt_instance_list = []
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

        y_list.append(name)

        y_pred_seg_argmax = torch.argmax(y_pred_seg, 1)  # should give shape (B, H, W)
        for img in y_pred_seg_argmax:
            img_name = './SavedImages/output_segmentation_%d.png' % img_ind
            save_image(img, img_name)
            img_ind += 1
            y_pred_seg_list.append(img_name)

        pred_instance_map = create_instance_maps(y_pred_seg, y_pred_center, y_pred_regression) # should give shape (H, W)

        img_name = './SavedImages/output_instances_%d.png' % img_ind
        save_image(pred_instance_map, img_name)
        img_ind += 1
        y_pred_instance_list.append(img_name)

    eval_main(y_pred_seg_list, y_list)
    eval_main2(y_pred_instance_list, y_list)

if __name__ == '__main__':
    max_epoch = 1000000
    min_loss = 0
    for i in './SavedModels':
        model_info = i.split('_')
        epoch = model_info[1]
        loss = model_info[2].split('.')
        if float(loss) < min_loss:
            min_loss = loss
        if epoch > max_epoch:
            max_epoch = epoch
    model = Model2('Model2', 'SimpleSegmentation/')
    model.load_state_dict(torch.load(os.path.join(config.save_dir, 'model_{}_{:.4f}.pth'.format(max_epoch, min_loss)))['state_dict'])
    val_dataset = get_cityscapes_dataset('~/SimpleSegmentation/CityscapesData', False, download=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    print('Finished inference.')

