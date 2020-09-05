import config
import torch
import torch.nn.functional as F
from torch.autograd.variable import Variable
import numpy as np
from dataloader import DataLoader, get_cityscapes_dataset, custom_collate
import torch.nn as nn
import torch.optim as optim
from deeplabv3 import Model, Model2, Model3, CapsuleModel, CapsuleModel2, CapsuleModel3, CapsuleModel4
from modelNew import CapsuleModelNew1
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
        image, (y_gt_seg, y_gt_center, y_gt_regression, y_gt_reg_pres, segmentation_weights), gt_class_list, gt_point_list, img_name = sample
        image = Variable(image.type(torch.FloatTensor))
        y_gt_seg = Variable(y_gt_seg.type(torch.LongTensor))
        y_gt_center = Variable(y_gt_center.type(torch.FloatTensor))
        y_gt_regression = Variable(y_gt_regression.type(torch.FloatTensor))
        y_gt_reg_pres = Variable(y_gt_reg_pres.type(torch.FloatTensor))
        segmentation_weights = Variable(segmentation_weights.type(torch.FloatTensor))

        if config.use_cuda:
            image = image.cuda()
            y_gt_seg = y_gt_seg.cuda()
            y_gt_center = y_gt_center.cuda()
            y_gt_regression = y_gt_regression.cuda()
            y_gt_reg_pres = y_gt_reg_pres.cuda()
            segmentation_weights = segmentation_weights.cuda()
            gt_class_list = [i.cuda() if len(i) != 0 else [] for i in gt_class_list]

        iteration += 1
        if config.poly_lr_scheduler:
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.learning_rate * (1 - (iteration / config.n_iterations) ** 0.9)

        optimizer.zero_grad()
        loss_list = []

        # y_pred_seg, y_pred_center, y_pred_regression = model(image)
        # y_pred_seg, y_pred_center, y_pred_regression, pred_class_list = model(image, gt_point_list, y_gt_seg) # if using CapsuleModel2
        y_pred_seg, y_pred_center, y_pred_regression, pred_class_list, y_pred_inst_maps, y_pred_segmentation_lists = model(image, gt_point_list, y_gt_reg_pres)

        # loss = (criterion1(y_pred_seg, y_gt_seg.squeeze(1)) * segmentation_weights).mean() * config.seg_coef  # may need to be segmentation_weights.squeeze(1)

        loss = (criterion4(y_pred_seg, y_gt_reg_pres) * segmentation_weights).mean() * config.seg_coef
        loss_list.append((criterion4(y_pred_seg, y_gt_reg_pres) * segmentation_weights).mean() * config.seg_coef)

        # loops through the ground-truth class_list and the class_outputs and adds the loss for each sample
        for j in range(len(gt_class_list)):
          if len(gt_class_list[j]) > 0:
              gt_class_onehot = F.one_hot(gt_class_list[j], config.n_classes)
              loss += criterion1(pred_class_list[j], gt_class_onehot.float()).mean() * config.class_coef
              loss_list.append(criterion1(pred_class_list[j], gt_class_onehot.float()).mean() * config.class_coef)        
    
        if config.use_instance:
            loss += criterion2(y_pred_center, y_gt_center) * config.center_coef
            loss += ((criterion3(y_pred_regression, y_gt_regression)) * y_gt_reg_pres).mean() * config.regression_coef
            loss_list.append(((criterion3(y_pred_regression, y_gt_regression)) * y_gt_reg_pres).mean() * config.regression_coef)

        acc = get_accuracy(y_pred_seg, y_gt_seg)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        accs.append(acc.item())

        del image, y_gt_seg, y_gt_center, y_gt_regression, y_gt_reg_pres, loss, acc, y_pred_seg, y_pred_center, y_pred_regression

        if (i + 1) % 10 == 0:
            print('Finished training %d batches. Loss: %.4f. Accuracy: %.4f.' % (i + 1, float(np.mean(losses)), float(np.mean(accs))), flush=True)
            # print('loss_list:', loss_list)

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
    if config.model == 'CapsuleModelNew1':
        model = CapsuleModelNew1('CapsuleModelNew1', 'SimpleSegmentation/')
        criterion1 = FocalLoss(alpha=0.25, gamma=2)
    elif config.model == 'CapsuleModel3':
        model = CapsuleModel3('CapsuleModel3', 'SimpleSegmentation/')
        criterion1 = FocalLoss(alpha=0.25, gamma=2)
    elif config.model == 'CapsuleModel4':
        model = CapsuleModel4('CapsuleModel4', 'SimpleSegmentation/')
        criterion1 = FocalLoss(alpha=0.25, gamma=2)
    else:
        model = Model3('Model3', 'SimpleSegmentation/')
        criterion1 = nn.CrossEntropyLoss(reduction='none', ignore_index=255)

    criterion2 = nn.MSELoss(reduction='mean')
    criterion3 = nn.L1Loss(reduction='none')
    criterion4 = nn.BCELoss(reduction='none')

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    tr_dataset = get_cityscapes_dataset(config.data_dir, True)

    if config.start_iteration != 0:
        save_file_path = os.path.join(config.save_dir, 'model_iteration_{}.pth'.format(config.start_iteration))
        if not os.path.exists(save_file_path):
            print('Save for iteration %d does not exist.' % config.start_iteration)
            exit()

        print('Loaded from: ', save_file_path)
        model.load_state_dict(torch.load(save_file_path)['state_dict'])

    iteration = config.start_iteration
    while iteration < config.n_iterations:
        tr_dataloader = DataLoader(tr_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=custom_collate)

        losses, _, iteration = train(model, tr_dataloader, criterion1, criterion2, criterion3, criterion4, optimizer, iteration)

    print('Training Finished')


if __name__ == '__main__':
    run_experiment()
    # inference.main()
    print()
