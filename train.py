import config
import torch
from torch.autograd.variable import Variable
import numpy as np
from dataloader import DataLoader, get_cityscapes_dataset
import torch.nn as nn
import torch.optim as optim
from deeplabv3 import DeepLabV3, Model2, Model3, Model4
import os


def get_accuracy(y_pred, y):
    y_argmax = torch.argmax(y_pred, 1)

    return torch.mean((y_argmax.long()==y.long()).type(torch.float))


def train(model, data_loader, criterion1, criterion2, criterion3, optimizer, iteration):
    model.train()

    if config.use_cuda:
        model.cuda()

    losses, accs = [], []

    for i, sample in enumerate(data_loader):
        image, (y_gt_seg, y_gt_center, y_gt_regression, y_gt_reg_pres, segmentation_weights), img_name = sample
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
        
        iteration +=1
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001 * (1 - (iteration/60000)**0.9)

        optimizer.zero_grad()
        y_pred_seg, y_pred_center, y_pred_regression = model(image)

        loss = (criterion1(y_pred_seg, y_gt_seg.squeeze(1)) * segmentation_weights).mean() * config.seg_coef  # may need to be segmentation_weights.squeeze(1)

        if config.use_instance:
            loss += criterion2(y_pred_center, y_gt_center)*config.center_coef
            loss += (criterion3(y_pred_regression, y_gt_regression)*y_gt_reg_pres).mean()*config.regression_coef

        acc = get_accuracy(y_pred_seg, y_gt_seg)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        accs.append(acc.item())

        del image, y_gt_seg, y_gt_center, y_gt_regression, y_gt_reg_pres, loss, acc, y_pred_seg, y_pred_center, y_pred_regression

        if (i + 1) % 10 == 0:
            print('Finished training %d batches. Loss: %.4f. Accuracy: %.4f.' % (i + 1, float(np.mean(losses)), float(np.mean(accs))), flush=True)

        if iteration % 1000 == 0:
            print('Model Saving.')

            save_file_path = os.path.join(config.save_dir2, 'model_iteration_{}.pth'.format(iteration))
            states = {
                      'iteration': iteration,
                      'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict()
                     }
            try:
                os.mkdir(config.save_dir2)
            except:
                pass
            torch.save(states, save_file_path)
            print('Model saved ', str(save_file_path))

    print('Finished training. Loss: %.4f. Accuracy: %.4f.' % (float(np.mean(losses)), float(np.mean(accs))))

    return float(np.mean(losses)), float(np.mean(accs)), iteration


def validation(model, data_loader, criterion1, criterion2, criterion3):
    model.eval()

    if config.use_cuda:
        model.cuda()

    losses, accs = [], []
    for i, sample in enumerate(data_loader):
        image, (y_gt_seg, y_gt_center, y_gt_regression, y_gt_reg_pres), img_name = sample
        y_gt_seg = Variable(y_gt_seg.type(torch.LongTensor))
        y_gt_center = Variable(y_gt_center.type(torch.FloatTensor))
        y_gt_regression = Variable(y_gt_regression.type(torch.FloatTensor))
        y_gt_reg_pres = Variable(y_gt_reg_pres.type(torch.FloatTensor))

        if config.use_cuda:
            image = image.cuda()
            y_gt_seg = y_gt_seg.cuda()
            y_gt_center = y_gt_center.cuda()
            y_gt_regression = y_gt_regression.cuda()
            y_gt_reg_pres = y_gt_reg_pres.cuda()

        with torch.no_grad():
            y_pred_seg, y_pred_center, y_pred_regression = model(image)

            loss = criterion1(y_pred_seg, y_gt_seg.squeeze(1))*config.seg_coef

            if config.use_instance:
                loss += criterion2(y_pred_center, y_gt_center)*config.center_coef
                loss += (criterion3(y_pred_regression, y_gt_regression)*y_gt_reg_pres).mean()*config.regression_coef

            acc = get_accuracy(y_pred_seg, y_gt_seg)

            losses.append(loss.item())
            accs.append(acc.item())

            del image, y_gt_seg, y_gt_center, y_gt_regression, y_gt_reg_pres, loss, acc, y_pred_seg, y_pred_center, y_pred_regression

            if (i + 1) % 100 == 0:
                print('Finished validating %d batches. Loss: %.4f. Accuracy: %.4f.' % (i+1, float(np.mean(losses)), float(np.mean(accs))))

    print('Finished validation. Loss: %.4f. Accuracy: %.4f.' % (float(np.mean(losses)), float(np.mean(accs))))

    return float(np.mean(losses)), float(np.mean(accs))


def run_experiment():
    # model = DeepLabV3('Model1', 'SimpleSegmentation/')
    model = Model4('Model4', 'SimpleSegmentation/')

    criterion1 = nn.CrossEntropyLoss(reduction='none', ignore_index=255)
    criterion2 = nn.MSELoss(reduction='mean')
    criterion3 = nn.L1Loss(reduction='none')

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0)

    tr_dataset = get_cityscapes_dataset(config.data_dir, True, download=True)
    val_dataset = get_cityscapes_dataset(config.data_dir, False, download=True)

    best_loss = 1000000

    if config.start_epoch != 1:
        max_epoch = 0
        for filename in os.listdir('./SavedModels/Run%d/' % config.model_id):
            model_info = filename.split('_')
            epoch = model_info[1]
            loss = model_info[2].split('.p')[0]
            if int(epoch) > max_epoch and int(epoch) < config.start_epoch:
                max_epoch = int(epoch)
                max_epoch_loss = float(loss)
        print('Loaded from: model_{}_{:.4f}.pth'.format(max_epoch, max_epoch_loss))
        model.load_state_dict(torch.load(os.path.join(config.save_dir, 'model_{}_{:.4f}.pth'.format(max_epoch, max_epoch_loss)))['state_dict'])    
    iteration = 0
    for epoch in range(config.start_epoch, config.n_epochs + 1):
        print('Epoch', epoch)

        tr_dataloader = DataLoader(tr_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

        losses, _, iteration = train(model, tr_dataloader, criterion1, criterion2, criterion3, optimizer, iteration)

        #losses, _ = validation(model, val_dataloader, criterion1, criterion2, criterion3)

        if epoch % 5 == 0:
            print('Model Improved -- Saving.')
            if losses < best_loss:
                best_loss = losses

            save_file_path = os.path.join(config.save_dir, 'model_{}_{:.4f}.pth'.format(epoch, losses))
            states = {
                        'epoch': epoch + 1,
                      'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict()
                     }
            try:
                os.mkdir(config.save_dir)
            except:
                pass
            torch.save(states, save_file_path)
            print('Model saved ', str(save_file_path))
    print('Training Finished')


if __name__ == '__main__':
    run_experiment()                                                                  
