from createInstanceMaps import create_instance_maps
import config
import numpy as np
import torch
from train import get_accuracy
from torch.autograd.variable import Variable
import os
from dataloader import DataLoader, get_cityscapes_dataset
from deeplabv3 import DeepLabV3, Model2
from PIL import Image


def mkdir(dir_name):
    if os.path.isdir(dir_name):
        return
    else:
        print('Directory %s has been created.' % dir_name)
        os.mkdir(dir_name)


def inference(model, data_loader):
    model.eval()

    if config.use_cuda:
        model.cuda()

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

        y_pred_seg_argmax = y_pred_seg_argmax.data.cpu().numpy()
        y_pred_center = y_pred_center.data.cpu().numpy()
        y_pred_regression = y_pred_regression.data.cpu().numpy()

        for j in range(len(y_pred_seg_argmax)):
            img_name_split = name[j].split('/')
            city = img_name_split[-2]
            mkdir('./SavedImages/val/Pixel/' + city)

            img_name = './SavedImages/val/Pixel/' + city + '/' + img_name_split[-1].replace('leftImg8bit', 'predFine_labelids')

            img = Image.fromarray(y_pred_seg_argmax[j].astype(np.uint8), mode='P')  # Converts numpy array to PIL Image
            img = img.resize(size=(2048, 1024), resample=Image.NEAREST)  # Resizes image
            img.save(img_name, "PNG", mode='P')  # Saves image

            pred_instance_map = create_instance_maps(y_pred_seg_argmax[j], y_pred_center[j], y_pred_regression[j])

            mkdir('./SavedImages/val/Instance/' + city)

            img_name = './SavedImages/val/Instance/' + city + '/' + img_name_split[-1].replace('leftImg8bit', 'predFine_instanceids')

            img = Image.fromarray(pred_instance_map, mode='L')  # Converts numpy array to PIL Image
            img = img.resize(size=(2048, 1024), resample=Image.NEAREST)  # Resizes image
            img.save(img_name, 'PNG', mode='L')  # Saves image

        if (i+1) % 5 == 0:
            print('Finished %d batches' % (i+1), flush=True)


def main():
    mkdir('./SavedImages/')
    mkdir('./SavedImages/val/')
    mkdir('./SavedImages/val/Pixel/')
    mkdir('./SavedImages/val/Instance/')

    # model_15_32.0749.pth

    epoch = 15
    loss = 32.0749

    model = Model2('Model2', 'SimpleSegmentation/')
    model.load_state_dict(torch.load(os.path.join(config.save_dir, 'model_{}_{:.4f}.pth'.format(epoch, loss)))['state_dict'])

    val_dataset = get_cityscapes_dataset(config.data_dir, False, download=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    inference(model, val_dataloader)


if __name__ == '__main__':
    main()
