import config
import torch
from torch.autograd.variable import Variable
import numpy as np
from dataloader import DataLoader, get_cityscapes_dataset
import torch.nn as nn
import torch.optim as optim
from deeplabv3 import DeepLabV3, Model2
import os


def get_accuracy(y_pred, y):
    y_argmax = torch.argmax(y_pred, 1)

    return torch.mean((y_argmax.long()==y.long()).type(torch.float))


def train(model, data_loader, criterion1, criterion2, optimizer):
    model.train()

    if config.use_cuda:
        model.cuda()

    losses, accs = [], []

    for i, sample in enumerate(data_loader):
        image, (y_gt_seg, y_gt_center, y_gt_regression, y_gt_reg_pres), img_name = sample
        image = Variable(image.type(torch.FloatTensor))
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

        optimizer.zero_grad()
        y_pred_seg, y_pred_center, y_pred_regression = model(image)

        loss = criterion1(y_pred_seg, y_gt_seg.squeeze(1))
        loss += criterion2(y_pred_center, y_gt_center).mean()
        loss += (criterion2(y_pred_regression, y_gt_regression)*y_gt_reg_pres).mean()

        acc = get_accuracy(y_pred_seg, y_gt_seg)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        accs.append(acc.item())

        if (i + 1) % 10 == 0:
            print('Finished training %d batches. Loss: %.4f. Accuracy: %.4f.' % (i + 1, float(np.mean(losses)), float(np.mean(accs))), flush=True)

    print('Finished training. Loss: %.4f. Accuracy: %.4f.' % (float(np.mean(losses)), float(np.mean(accs))))

    return float(np.mean(losses)), float(np.mean(accs))


def validation(model, data_loader, criterion1, criterion2):
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

        y_pred_seg, y_pred_center, y_pred_regression = model(image)
        loss = criterion1(y_pred_seg, y_gt_seg.squeeze(1))
        loss += criterion2(y_pred_center, y_gt_center).mean()
        loss += (criterion2(y_pred_regression, y_gt_regression)*y_gt_reg_pres).mean()

        acc = get_accuracy(y_pred_seg, y_gt_seg)

        losses.append(loss.item())
        accs.append(acc.item())

        if (i + 1) % 100 == 0:
            print('Finished validating %d batches. Loss: %.4f. Accuracy: %.4f.' % (i+1, float(np.mean(losses)), float(np.mean(accs))))
    print('Finished validation. Loss: %.4f. Accuracy: %.4f.' % (float(np.mean(losses)), float(np.mean(accs))))

    return float(np.mean(losses)), float(np.mean(accs))


def run_experiment():
    # model = DeepLabV3('Model1', 'SimpleSegmentation/')
    model = Model2('Model2', 'SimpleSegmentation/')
    if config.start_epoch != 1:
        max_epoch = 0
        min_loss = 10000000.0000
        for filename in os.listdir('./SavedModels/Run%d/' % config.model_id):
            model_info = filename.split('_')
            epoch = model_info[1]
            loss = model_info[2].split('.p')[0]
            if float(loss) < min_loss and int(epoch) < config.start_epoch:
                min_loss = float(loss)
            if int(epoch) > max_epoch and int(epoch) < config.start_epoch:
                max_epoch = int(epoch)
        print('Loaded from: model_{}_{:.4f}.pth'.format(max_epoch, min_loss))
        model.load_state_dict(torch.load(os.path.join(config.save_dir, 'model_{}_{:.4f}.pth'.format(max_epoch, min_loss)))['state_dict'])
    
    criterion1 = nn.CrossEntropyLoss(reduction='mean')
    criterion2 = nn.MSELoss(reduction='none')

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-7)

    tr_dataset = get_cityscapes_dataset(config.data_dir, True, download=True)
    val_dataset = get_cityscapes_dataset(config.data_dir, False, download=True)

    best_loss = 1000000

    for epoch in range(1, config.n_epochs + 1):
        print('Epoch', epoch)

        tr_dataloader = DataLoader(tr_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        losses, _ = train(model, tr_dataloader, criterion1, criterion2, optimizer)

        losses, _ = validation(model, val_dataloader, criterion1, criterion2)

        if losses < best_loss:
            print('Model Improved -- Saving.')
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
    
                                                                                   
