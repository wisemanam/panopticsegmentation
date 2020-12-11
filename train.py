import config
import torch
import torch.nn.functional as F
from torch.autograd.variable import Variable
import numpy as np
from dataloader import DataLoader, get_cityscapes_dataset, custom_collate, get_coco_dataset
import torch.nn as nn
import torch.optim as optim
from modelNew import CapsuleModel5, CapsuleModel4, NewCapsuleModel6, CapsuleModel7
import os
from losses import MarginLoss
from focal import FocalLoss

def get_accuracy(y_pred, y):
    y_argmax = torch.argmax(y_pred, 1)

    return torch.mean((y_argmax.long() == y.long()).type(torch.float))


def train(model, data_loader, criterion1, criterion2, criterion3, criterion4, optimizer, iteration):
    model.train()

    if config.use_cuda:
        model.cuda()

    losses, accs = [], []

    for i, sample in enumerate(data_loader):
        image, (y_gt_regression, y_gt_fgbg_seg, segmentation_weights), gt_class_list, gt_point_list, img_name = sample
        image = Variable(image.type(torch.FloatTensor))
        y_gt_regression = Variable(y_gt_regression.type(torch.FloatTensor))
        y_gt_fgbg_seg = Variable(y_gt_fgbg_seg.type(torch.FloatTensor))
        segmentation_weights = Variable(segmentation_weights.type(torch.FloatTensor))

        if config.use_cuda:
            image = image.cuda()
            y_gt_regression = y_gt_regression.cuda()
            y_gt_fgbg_seg = y_gt_fgbg_seg.cuda()
            segmentation_weights = segmentation_weights.cuda()
            gt_class_list = [i.cuda() if len(i) != 0 else [] for i in gt_class_list]

        iteration += 1
        if config.poly_lr_scheduler:
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.learning_rate * (1 - (iteration / config.n_iterations) ** 0.9)

        optimizer.zero_grad()
        classification_loss = 0

        if config.model == 'CapsuleModel7':
            y_pred_fgbg_seg, y_pred_regression, pred_class_list, y_pred_inst_maps, y_pred_segmentation_lists, y_dense_class_list = model(image, gt_point_list, y_gt_fgbg_seg, two_stage=False)
        else:
            y_pred_fgbg_seg, y_pred_regression, pred_class_list, y_pred_inst_maps, y_pred_segmentation_lists = model(image, gt_point_list, y_gt_fgbg_seg)

        for pred_fgbg in y_pred_fgbg_seg:
            loss = (criterion4(pred_fgbg, y_gt_fgbg_seg) * segmentation_weights).mean() * config.seg_coef

        if config.model != 'CapsuleModel7':
            # loops through the ground-truth class_list and the class_outputs and adds the loss for each sample
            for pred_class in pred_class_list:
                for j in range(len(gt_class_list)):
                    if len(gt_class_list[j]) > 0:
                        gt_class_onehot = F.one_hot(gt_class_list[j], config.n_classes)
                        loss += criterion1(pred_class[j], gt_class_onehot.float()).mean() * config.class_coef
                        classification_loss = criterion1(pred_class[j], gt_class_onehot.float()).mean() * config.class_coef

        else:
            for m, pred_class in enumerate(pred_class_list):
                pred_class_dense = y_dense_class_list[m]
                for j in range(len(gt_class_list)):
                    if len(gt_class_list[j]) > 0:
                        gt_class_onehot = F.one_hot(gt_class_list[j], config.n_classes)
                        loss += criterion1(pred_class[j], gt_class_onehot.float()).mean() * config.class_coef
                        classification_loss = criterion1(pred_class[j], gt_class_onehot.float()).mean() * config.class_coef
                        for k in range(len(gt_class_list[j])):
                            gt_class_onehot_k = gt_class_onehot[k]
                            pred_class_dense_k = pred_class_dense[j][k]
                            gt_class_onehot_k = gt_class_onehot_k.unsqueeze(0).repeat(pred_class_dense_k.shape[0], 1)
                            loss += criterion1(pred_class_dense_k, gt_class_onehot_k.float()).mean() * config.class_coef
      

        if config.use_instance:
            for pred_reg in y_pred_regression:
                loss += ((criterion3(pred_reg, y_gt_regression)) * y_gt_fgbg_seg).mean() * config.regression_coef

        acc = get_accuracy(y_pred_fgbg_seg[-1], y_gt_fgbg_seg) # y_gt_seg)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        accs.append(acc.item())

        del image, y_gt_regression, y_gt_fgbg_seg, loss, acc, y_pred_fgbg_seg, y_pred_regression, segmentation_weights, gt_class_list, gt_point_list

        if (i + 1) % 10 == 0:
            print('Finished training %d batches. Loss: %.4f. Accuracy: %.4f.' % (i + 1, float(np.mean(losses)), float(np.mean(accs))), flush=True)

        if iteration % config.save_every_n_iters == 0:
            print('Model Saving.')

            save_file_path = os.path.join(config.save_dir, 'model_iteration_{}.pth'.format(iteration))
            states = {
                'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            try:
                os.mkdir(config.save_dir)
            except:
                pass
            torch.save(states, save_file_path)
            print('Model saved ', str(save_file_path))

    print('Finished training %d iterations. Loss: %.4f. Accuracy: %.4f.' % (iteration, float(np.mean(losses)), float(np.mean(accs))))

    return float(np.mean(losses)), float(np.mean(accs)), iteration


def run_experiment():
    if config.model == 'CapsuleModel5':
        model = CapsuleModel5('CapsuleModel5', 'SimpleSegmentation/')
        criterion1 = FocalLoss(alpha=0.25, gamma=2)
    elif config.model == 'CapsuleModel4':
        model = CapsuleModel4('CapsuleModel4', 'SimpleSegmentation/')
        criterion1 = FocalLoss(alpha=0.25, gamma=2)
    elif config.model == 'NewCapsuleModel6':
        model = NewCapsuleModel6('NewCapsuleModel6', 'SimpleSegmentation/')
        criterion1 = FocalLoss(alpha=0.25, gamma=2)
    elif config.model == 'CapsuleModel7':
        model = CapsuleModel7('CapsuleModel7', 'SimpleSegmentation/')
        criterion1 = FocalLoss(alpha=0.25, gamma=2)
    # model = nn.DataParallel(model)

    criterion2 = nn.MSELoss(reduction='mean')
    criterion3 = nn.L1Loss(reduction='none')
    criterion4 = nn.BCELoss(reduction='none')

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    tr_dataset = get_coco_dataset(config.data_dir + '/images/train2017', True)

    if config.start_iteration != 0:
        save_file_path = os.path.join(config.save_dir, 'model_iteration_{}.pth'.format(config.start_iteration))
        if not os.path.exists(save_file_path):
            print('Save for iteration %d does not exist.' % config.start_iteration)
            exit()

        print('Loaded from: ', save_file_path)
        saved_weights = torch.load(save_file_path)['state_dict']
        model.load_state_dict(saved_weights)
        saved_weights.clear()

    iteration = config.start_iteration
    while iteration < config.n_iterations:
        tr_dataloader = DataLoader(tr_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=custom_collate)

        losses, _, iteration = train(model, tr_dataloader, criterion1, criterion2, criterion3, criterion4, optimizer, iteration)

    print('Training Finished')


if __name__ == '__main__':
    run_experiment()
