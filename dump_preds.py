import config
import numpy as np
import torch
from train import get_accuracy
from torch.autograd.variable import Variable
import os
from dataloader import DataLoader, get_cityscapes_dataset, custom_collate, get_coco_dataset
from PIL import Image
from modelNew import CapsuleModel5, CapsuleModel6, NewCapsuleModel6, CapsuleModel7, CapsuleModel4

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
        image, (y_gt_regression, y_gt_seg, segmentation_weights), gt_class_list, gt_point_list, name = sample

        if config.use_cuda:
            image = image.cuda()
            y_gt_seg = y_gt_seg.cuda()
            y_gt_regression = y_gt_regression.cuda()

        with torch.no_grad():
            if config.model == 'CapsuleModel7':
                y_pred_seg, y_pred_regression, pred_class_list, inst_maps, y_pred_segmentation_lists, y_dense_class_list = model(image, None, None, None, two_stage=False)
            else:
                y_pred_seg, y_pred_regression, pred_class_list, inst_maps, y_pred_segmentation_lists = model(image, None, None, None)
            
            y_pred_seg = y_pred_seg[-1]
            y_pred_regression = y_pred_regression[-1]
            pred_class_list = pred_class_list[-1]
            inst_maps = inst_maps[-1]
            y_pred_segmentation_lists = y_pred_segmentation_lists[-1]

            y_gt_seg = y_gt_seg.data.cpu().numpy()
            y_gt_regression = y_gt_regression.data.cpu().numpy()
            y_pred_seg = y_pred_seg.data.cpu().numpy()
            y_pred_regression = y_pred_regression.data.cpu().numpy()

            if config.data_dir == './CityscapesData':
                for j in range(len(y_gt_seg)):
                    img_name_split = name[j].split('/')
                    city = img_name_split[-2]
                    mkdir('./DumpedOutputs/' + city)

                    img_name = './DumpedOutputs/' + city + '/' + img_name_split[-1].replace('_leftImg8bit.png', '')
                    np.save(img_name + '_gt_seg.npy', y_gt_seg[j])

                    img_name = './DumpedOutputs/' + city + '/' + img_name_split[-1].replace('_leftImg8bit.png', '')
                    np.save(img_name + '_gt_regression.npy', y_gt_regression[j])

                    img_name = './DumpedOutputs/' + city + '/' + img_name_split[-1].replace('_leftImg8bit.png', '')
                    np.save(img_name + '_pr_seg.npy', y_pred_seg[j])

                    img_name = './DumpedOutputs/' + city + '/' + img_name_split[-1].replace('_leftImg8bit.png', '')
                    np.save(img_name + '_pr_regression.npy', y_pred_regression[j])

                    if isinstance(inst_maps[j], list) == False:
                        img_name = './DumpedOutputs/' + city + '/' + img_name_split[-1].replace('_leftImg8bit.png', '')
                        np.save(img_name + '_pr_inst_maps.npy', inst_maps[j].cpu().numpy())
            elif config.data_dir == './CocoData':
                for j in range(len(y_gt_seg)):
                    img_name_split = name
                    mkdir('./DumpedOutputs/')

                    img_name = './DumpedOutputs/' + img_name_split[-1].replace('_leftImg8bit.png', '')
                    np.save(img_name + '_gt_seg.npy', y_gt_seg[j])

                    img_name = './DumpedOutputs/' + img_name_split[-1].replace('_leftImg8bit.png', '')
                    np.save(img_name + '_gt_regression.npy', y_gt_regression[j])

                    img_name = './DumpedOutputs/' + img_name_split[-1].replace('_leftImg8bit.png', '')
                    np.save(img_name + '_pr_seg.npy', y_pred_seg[j])

                    img_name = './DumpedOutputs/' + img_name_split[-1].replace('_leftImg8bit.png', '')
                    np.save(img_name + '_pr_regression.npy', y_pred_regression[j])

                    if isinstance(inst_maps[j], list) == False:
                        img_name = './DumpedOutputs/' + img_name_split[-1].replace('_leftImg8bit.png', '')
                        np.save(img_name + '_pr_inst_maps.npy', inst_maps[j].cpu().numpy())

            n_samples += 1
            if n_samples == 100:
                exit()

        if (i+1) % 5 == 0:
            print('Finished %d batches' % (i+1), flush=True)


def main():
    mkdir('./DumpedOutputs/')

    iteration = 63000

    if config.model == 'CapsuleModel5':
        model = CapsuleModel5('CapsuleModel5', 'SimpleSegmentation/')
    elif config.model == 'CapsuleModel4':
        model = CapsuleModel4('CapsuleModel4', 'SimpleSegmentation/')
    elif config.model == 'NewCapsuleModel6':
        model = NewCapsuleModel6('NewCapsuleModel6', 'SimpleSegmentation/')
    elif config.model == 'CapsuleModel7':
        model = CapsuleModel7('CapsuleModel7', 'SimpleSegmentation/')

    model.load_state_dict(torch.load(os.path.join(config.save_dir, 'model_iteration_{}.pth'.format(iteration)))['state_dict'])

    val_dataset = get_coco_dataset(config.data_dir + '/images/val2017', False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=custom_collate)

    inference(model, val_dataloader)



if __name__ == '__main__':
    main()
