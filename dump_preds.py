import config
import numpy as np
import torch
from train import get_accuracy
from torch.autograd.variable import Variable
import os
from dataloader import DataLoader, get_cityscapes_dataset, custom_collate
from deeplabv3 import Model, CapsuleModel, Model2, Model3, CapsuleModel2
from PIL import Image

def mkdir(dir_name):
    if os.path.isdir(dir_name):
        return
    else:
        print('Directory %s has been created.' % dir_name, flush=True)
        os.mkdir(dir_name)


def convert_train_id_to_eval_id(prediction):
    # Shape (B, C, H, W)
    b, c, h, w = prediction.shape
    eval_preds = torch.zeros((b, 34, h, w), dtype=prediction.dtype)  # TODO check if this should be zero or -infinity

    _CITYSCAPES_TRAIN_ID_TO_EVAL_ID = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

    eval_preds[:, _CITYSCAPES_TRAIN_ID_TO_EVAL_ID] = prediction

    return eval_preds


def inference(model, data_loader):
    model.eval()

    if config.use_cuda:
        model.cuda()

    n_samples = 0
    for i, sample in enumerate(data_loader):
        # image, (y_gt_seg, y_gt_center, y_gt_regression, y_gt_reg_pres, _), name = sample
        image, (y_gt_seg, y_gt_center, y_gt_regression, y_gt_reg_pres, _), gt_class_list, gt_point_list, name = sample

        if config.use_cuda:
            image = image.cuda()
            y_gt_seg = y_gt_seg.cuda()
            y_gt_center = y_gt_center.cuda()
            y_gt_regression = y_gt_regression.cuda()

        with torch.no_grad():
            # y_pred_seg, y_pred_center, y_pred_regression = model(image)
            # y_pred_seg, y_pred_center, y_pred_regression, pred_class_list = model(image, gt_point_list, y_gt_seg) # if using CapsuleModel2
            y_pred_seg, y_pred_center, y_pred_regression, pred_class_list, inst_maps, y_pred_segmentation_lists = model(image, gt_point_list, y_gt_seg)

            y_gt_seg = y_gt_seg.data.cpu().numpy()
            y_gt_center = y_gt_center.data.cpu().numpy()
            y_gt_regression = y_gt_regression.data.cpu().numpy()
            y_pred_seg = y_pred_seg.data.cpu().numpy()
            y_pred_center = y_pred_center.data.cpu().numpy()
            y_pred_regression = y_pred_regression.data.cpu().numpy()

            for j in range(len(y_gt_seg)):
                img_name_split = name[j].split('/')
                city = img_name_split[-2]
                mkdir('./DumpedOutputs/' + city)

                img_name = './DumpedOutputs/' + city + '/' + img_name_split[-1].replace('_leftImg8bit.png', '')
                np.save(img_name + '_gt_seg.npy', y_gt_seg[j])

                img_name = './DumpedOutputs/' + city + '/' + img_name_split[-1].replace('_leftImg8bit.png', '')
                np.save(img_name + '_gt_center.npy', y_gt_center[j])

                img_name = './DumpedOutputs/' + city + '/' + img_name_split[-1].replace('_leftImg8bit.png', '')
                np.save(img_name + '_gt_regression.npy', y_gt_regression[j])

                img_name = './DumpedOutputs/' + city + '/' + img_name_split[-1].replace('_leftImg8bit.png', '')
                np.save(img_name + '_pr_seg.npy', y_pred_seg[j])

                img_name = './DumpedOutputs/' + city + '/' + img_name_split[-1].replace('_leftImg8bit.png', '')
                np.save(img_name + '_pr_center.npy', y_pred_center[j])

                img_name = './DumpedOutputs/' + city + '/' + img_name_split[-1].replace('_leftImg8bit.png', '')
                np.save(img_name + '_pr_regression.npy', y_pred_regression[j])

                img_name = './DumpedOutputs/' + city + '/' + img_name_split[-1].replace('_leftImg8bit.png', '')
                np.save(img_name + '_pr_inst_maps.npy', inst_maps[j])

                n_samples += 1
                if n_samples == 100:
                    exit()

        if (i+1) % 5 == 0:
            print('Finished %d batches' % (i+1), flush=True)


def main():
    mkdir('./DumpedOutputs/')

    # model_60_0.4552.pth model_50_20.2748.pth

    iteration = 20000

    # model_10_1.2785.pth
    if config.model == 'CapsuleModel':
        model = CapsuleModel2('CapsuleModel2', 'SimpleSegmentation/')
    else:
        model = Model3('Model3', 'SimpleSegmentation/')

    model.load_state_dict(torch.load(os.path.join(config.save_dir, 'model_iteration_{}.pth'.format(iteration)))['state_dict'])

    val_dataset = get_cityscapes_dataset(config.data_dir, False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=custom_collate)

    inference(model, val_dataloader)



if __name__ == '__main__':
    main()
