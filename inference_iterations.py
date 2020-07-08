
from postprocessing import PostProcessing
import config
import numpy as np
import torch
import os
from dataloader import DataLoader, get_cityscapes_dataset
from deeplabv3 import DeepLabV3, Model2, Model3, Model4
from PIL import Image


def mkdir(dir_name):
    if os.path.isdir(dir_name):
        return
    else:
        print('Directory %s has been created.' % dir_name, flush=True)
        os.mkdir(dir_name)


def inference(model, data_loader):
    model.eval()
    post = PostProcessing()

    if config.use_cuda:
        model.cuda()
        post.cuda()

    for i, sample in enumerate(data_loader):
        image, (y_gt_seg, y_gt_center, y_gt_regression, y_gt_reg_pres, segmentation_weights), name = sample

        if config.use_cuda:
            image = image.cuda()
            y_gt_seg = y_gt_seg.cuda()
            y_gt_center = y_gt_center.cuda()
            y_gt_regression = y_gt_regression.cuda()
            y_gt_reg_pres = y_gt_reg_pres.cuda()
            segmentation_weights = segmentation_weights.cuda()

        with torch.no_grad():
            y_pred_seg, y_pred_center, y_pred_regression = model(image)

            instance_map_outputs, y_pred_seg_argmax = post(y_pred_seg, y_pred_center, y_pred_regression)

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
    # print(torch.__version__)
    # exit()
    mkdir('./SavedImages/')
    mkdir('./SavedImages/val/')
    mkdir('./SavedImages/val/Pixel/')
    mkdir('./SavedImages/val/Instance/')

    model = Model4('Model4', 'SimpleSegmentation/')

    max_iteration = 0
    for filename in os.listdir('./SavedIterations/Run%d/' % config.model_id):
        model_info = filename.split('_')
        iteration = model_info[2].split('.')[0]
        if int(iteration) > max_iteration:
            max_iteration = int(iteration)
    model.load_state_dict(torch.load(os.path.join(config.save_dir2, 'model_iteration_{}.pth'.format(iteration)))['state_dict'])

    val_dataset = get_cityscapes_dataset(config.data_dir, False, download=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    inference(model, val_dataloader)


if __name__ == '__main__':
    main()

