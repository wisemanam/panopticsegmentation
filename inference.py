import config
import numpy as np
import torch
from torch import nn
import os
from dataloader import DataLoader, get_cityscapes_dataset, custom_collate
from modelNew import CapsuleModel5, CapsuleModel6, NewCapsuleModel6, CapsuleModel7
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
        

def inference(model, data_loader):
    model.eval()
    convert_to_eval = Converter(dims=(config.h, config.w))

    if config.use_cuda:
        model.cuda()
        convert_to_eval.cuda()

    for i, sample in enumerate(data_loader):
        image, (y_gt_regression, y_gt_reg_pres, segmentation_weights), gt_class_list, gt_point_list, img_name = sample

        if config.use_cuda:
            image = image.cuda()
            y_gt_regression = y_gt_regression.cuda()
            y_gt_reg_pres = y_gt_reg_pres.cuda()

        with torch.no_grad():
            if config.model == 'CapsuleModel7':
                y_pred_fg_seg, y_pred_regressions, y_pred_class, inst_maps, segmentation_lists, y_dense_class_list = model(image, None, None, None, two_stage=False)
            else:
                y_pred_fg_seg, y_pred_regressions, y_pred_class, inst_maps, segmentation_lists = model(image, None, None, None, two_stage=True)

            y_pred_fg_seg = y_pred_fg_seg[-1]
            y_pred_regressions = y_pred_regressions[-1]
            y_pred_class = y_pred_class[-1]
            inst_maps = inst_maps[-1]
            segmentation_lists = segmentation_lists[-1]


            # if config.n_classes == 19:  # TODO implement the class conversion later
            #     y_pred_class = convert_to_eval(y_pred_class)


        for j in range(len(y_pred_fg_seg)):
            img_name_split = img_name[j].split('/')
            # city = img_name_split[-2]

            # mkdir('./SavedImages/val/Instance/' + city)
            # inst_dir_name = './SavedImages/val/Instance/' + city + '/' + img_name_split[-1].replace('leftImg8bit', '')[:-5] + '/'
            # mkdir(inst_dir_name)
            ('./SavedImages/val/Instance/')

            class_probs = y_pred_class[j]  # Shape (N, C)

            if len(class_probs) != 0:
                class_probs = class_probs.cpu()

                class_preds = np.argmax(class_probs, -1)

                segmentation_list = segmentation_lists[j]  # length N

            lines = []
            for inst in range(len(class_probs)):
                binary_map = segmentation_list[inst]
                inst_class = class_preds[inst]
                inst_prob = 1.0
                seg_prob = class_probs[inst, inst_class]

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
    mkdir('./SavedImages/')
    mkdir('./SavedImages/val/')
    mkdir('./SavedImages/val/Pixel/')
    mkdir('./SavedImages/val/Instance/')

    iteration = 90000

    if config.model == 'CapsuleModel5':
        model = CapsuleModel5('CapsuleModel5', 'SimpleSegmentation/')
    elif config.model == 'CapsuleModel6':
        model = CapsuleModel6('CapsuleModel6', 'SimpleSegmentation/')
    elif config.model == 'NewCapsuleModel6':
        model = NewCapsuleModel6('NewCapsuleModel6', 'SimpleSegmentation/')
    elif config.model == 'CapsuleModel7':
        model = CapsuleModel7('CapsuleModel7', 'SimpleSegmentation/')

    model.load_state_dict(torch.load(os.path.join(config.save_dir, 'model_iteration_{}.pth'.format(iteration)))['state_dict'])

    val_dataset = get_cityscapes_dataset(config.data_dir, False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=custom_collate)

    inference(model, val_dataloader)


if __name__ == '__main__':
    main()
