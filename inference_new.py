from postprocessing import PostProcessing, PostProcessing2, PostProcessing3
from postprocessing4 import PostProcessing4
import config
import numpy as np
import torch
from torch import nn
import os
from dataloader import DataLoader, get_cityscapes_dataset, custom_collate
from deeplabv3 import Model, CapsuleModel, Model2, CapsuleModel2
from PIL import Image


def mkdir(dir_name):
    if os.path.isdir(dir_name):
        return
    else:
        print('Directory %s has been created.' % dir_name, flush=True)
        os.mkdir(dir_name)


class Converter(nn.Module):
    def __init__(self, dims=(512, 1024)):
        super(Converter, self).__init__()

        self.zero_vals = -1000000.0
        h, w = dims
        eval_preds = torch.ones((1, 34, h, w), dtype=torch.float)

        self.register_buffer('eval_preds', eval_preds)

    def forward(self, prediction):
        # Shape (B, C, H, W)
        b, _, _, _ = prediction.shape
        eval_preds = (self.eval_preds.repeat((b, 1, 1, 1)))*self.zero_vals

        _CITYSCAPES_TRAIN_ID_TO_EVAL_ID = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

        eval_preds[:, _CITYSCAPES_TRAIN_ID_TO_EVAL_ID] = prediction

        return eval_preds

def get_points(instance_maps):
    # instance_maps is a list of N (h, w) tensors, N being the number of instances in the image
    point_lists = []
    for i, inst_map in enumerate(instance_maps):
        point_list = []
        unique_insts = torch.unique(inst_map)
        inst_map = inst_map.cpu()
        for inst in unique_insts:
            pixels = np.stack(np.where(inst_map == inst))
            point_list.append(torch.Tensor(pixels).long())
        point_lists.append(point_list)
    return point_lists

def inference(model, data_loader):
    model.eval()
    post = PostProcessing4(dims=(config.h, config.w), kernel_size=7, top_k=200, circle_radius=7)
    convert_to_eval = Converter(dims=(config.h, config.w))

    if config.use_cuda:
        model.cuda()
        post.cuda()
        convert_to_eval.cuda()

    for i, sample in enumerate(data_loader):
        image, (y_gt_seg, y_gt_center, y_gt_regression, y_gt_reg_pres, _), gt_class_list, gt_point_list, name = sample

        if config.use_cuda:
            image = image.cuda()
            y_gt_seg = y_gt_seg.cuda()
            y_gt_center = y_gt_center.cuda()
            y_gt_regression = y_gt_regression.cuda()
            y_gt_reg_pres = y_gt_reg_pres.cuda()
            gt_class_list = [i.cuda() if len(i) != 0 else [] for i in gt_class_list]

        with torch.no_grad():
            # y_pred_seg, y_pred_center, y_pred_regression = model(image)
            y_pred_seg, y_pred_center, y_pred_regression, pred_class_list = model(image, gt_point_list, y_gt_seg)

            if config.n_classes == 19:
                y_pred_seg = convert_to_eval(y_pred_seg)

            instance_map_outputs, y_pred_seg_argmax, instance_maps = post(y_pred_seg, y_pred_center, y_pred_regression, y_gt_seg.float())
            
            points = get_points(instance_maps)

            y_pred_seg, y_pred_center, y_pred_regression, pred_class_list = model(image, points, y_gt_seg)

        y_pred_seg_argmax = y_pred_seg_argmax.data.cpu().numpy()

        for j in range(len(y_pred_seg_argmax)):
            img_name_split = name[j].split('/')
            city = img_name_split[-2]
            mkdir('./SavedImages/val/Pixel/' + city)

            img_name = './SavedImages/val/Pixel/' + city + '/' + img_name_split[-1].replace('leftImg8bit', 'predFine_labelids')

            img = Image.fromarray(y_pred_seg_argmax[j].astype(np.uint8), mode='P')  # Converts numpy array to PIL Image
            img = img.resize(size=(2048, 1024), resample=Image.NEAREST)  # Resizes image
            img.save(img_name, "PNG", mode='P')  # Saves image

            mkdir('./SavedImages/val/Instance/' + city)
            inst_dir_name = './SavedImages/val/Instance/' + city + '/' + img_name_split[-1].replace('leftImg8bit', '')[:-5] + '/'
            mkdir(inst_dir_name)

            instance_map_output = instance_map_outputs[j]

            lines = []
            
            for inst, (binary_map, inst_class, inst_prob, seg_prob, n_pixels) in enumerate(instance_map_output):
                if n_pixels <= 0:
                    continue
                binary_map = binary_map.data.cpu().numpy()

                img = Image.fromarray(binary_map, mode='L')  # Converts numpy array to PIL Image
                img = img.resize(size=(2048, 1024), resample=Image.NEAREST)  # Resizes image
                img.save(inst_dir_name + 'instance%d.png' % inst, 'PNG', mode='L')  # Saves image

                line_str = (inst_dir_name + 'instance%d.png'% inst).replace('./SavedImages/val/Instance/', '')

                line_str = line_str + (' %d %.4f\n' % (inst_class, inst_prob*seg_prob))

                lines.append(line_str)

            file_dir_name = './SavedImages/val/Instance/' + img_name_split[-1].replace('leftImg8bit', '')[:-5] + '.txt'
            with open(file_dir_name, 'w') as f:
                f.writelines(lines)

        if (i+1) % 5 == 0:
            print('Finished %d batches' % (i+1), flush=True)


def main():
    #print(torch.__version__)
    #import torchvision
    #print(torchvision.__version__)
    #exit()

    mkdir('./SavedImages/')
    mkdir('./SavedImages/val/')
    mkdir('./SavedImages/val/Pixel/')
    mkdir('./SavedImages/val/Instance/')

    # model_45_0.4305.pth  model_60_0.4552.pth model_50_20.2748.pth

    iteration = 60000

    # model_10_1.2785.pth
    if config.model == 'CapsuleModel':
        model = CapsuleModel2('CapsuleModel2', 'SimpleSegmentation/')
    else:
        model = Model2('Model', 'SimpleSegmentation/')

    model.load_state_dict(torch.load(os.path.join(config.save_dir, 'model_iteration_{}.pth'.format(iteration)))['state_dict'])

    val_dataset = get_cityscapes_dataset(config.data_dir, False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=custom_collate)

    inference(model, val_dataloader)


if __name__ == '__main__':
    main()
