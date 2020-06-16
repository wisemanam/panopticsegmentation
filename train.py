import config
import torch
from torch.autograd.variable import Variable
import numpy as np
from dataloader import TrainDataset, ValidationDataset, DataLoader, get_cityscapes_dataset
import torch.nn as nn
import torch.optim as optim
from deeplabv3 import DeepLabV3
import os
import inference

def get_accuracy(y_pred, y):
    y_argmax = torch.argmax(y_pred, 1)

    return torch.mean((y_argmax.long()==y.long()).type(torch.float))



def train(model, data_loader, criterion, optimizer):
    model1.train()
    model2.train()
    
    if config.use_cuda:
        model1.cuda()
        model2.cuda()
        
    losses, accs = [], []
    for i, sample in enumerate(data_loader):
        (x, y), name = sample

        if config.use_cuda:
            x = x.cuda()
            y = y.cuda()

        optimizer.zero_grad()
        y = y*255.0
        y_pred_seg = model1(x)
        y_pred_center, y_pred_regression = model2(x)
        
        y_gt_seg = y
        image_name, (segmentation_map, y_gt_center, y_gt_regression) = data_loader.dataset.__getitem__(i)
        
        loss = criterion(y_pred, (y.long()).squeeze())
        loss += criterion2(y_pred_center, y_gt_center)
        loss += criterion2(y_pred_regression, y_gt_regression)

        acc = get_accuracy(y_pred, y)
        # Do we need to get the accuracy for instance segmentation?
        
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        accs.append(acc.item())

        if (i + 1) % 100 == 0:
            print('Finished training %d batches. Loss: %.4f. Accuracy: %.4f.' % (i+1, float(np.mean(losses)), float(np.mean(accs))))

    print('Finished training. Loss: %.4f. Accuracy: %.4f.' % (float(np.mean(losses)), float(np.mean(accs))))

    return float(np.mean(losses)), float(np.mean(accs))


def validation(model, data_loader, criterion):
    model1.eval()
    model2.eval()
    
    if config.use_cuda:
        model1.cuda()
        model2.cuda()

        losses, accs = [], []
        for i, sample in enumerate(data_loader):
            (x, y), name = sample

            if config.use_cuda:
                x = x.cuda()
                y = y.cuda()

            y=y*255.0
            y_pred_seg = model1(x)
            y_pred_center, y_pred_regression = model2(x)

            y_gt_seg = y
            image_name, (segmentation_map, y_gt_center, y_gt_regression) = data_loader.dataset.__getitem__(i)
            
            loss = criterion1(y_pred_seg, (y_gt_seg.long()).squeeze())
            loss += criterion2(y_pred_center, y_gt_center)
            loss += criterion2(y_pred_regression, y_gt_regression)
            
            acc = get_accuracy(y_pred_seg, y_gt_seg)

            losses.append(loss.item())
            accs.append(acc.item())

            if (i + 1) % 100 == 0:
                print('Finished validating %d batches. Loss: %.4f. Accuracy: %.4f.' % (i+1, float(np.mean(losses)), float(np.mean(accs))))
        print('Finished validation. Loss: %.4f. Accuracy: %.4f.' % (float(np.mean(losses)), float(np.mean(accs))))

        return float(np.mean(losses)), float(np.mean(accs))

def run_experiment():
    model = DeepLabV3('Model1', 'SimpleSegmentation/')
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-7)
    tr_dataset = get_cityscapes_dataset('~/SimpleSegmentation/CityscapesData', True, download=True)  # TrainDataset()  # A custom dataloader may be needed, in which case use TrainDataset()
    val_dataset = get_cityscapes_dataset('~/SimpleSegmentation/CityscapesData', False, download=True)  # ValidationDataset() # A custom dataloader may be needed, in which case use ValidationDataset()

    best_loss = 1000000

    for epoch in range(1, config.n_epochs + 1):
        print('Epoch', epoch)

        tr_dataloader = DataLoader(tr_dataset, batch_size=config.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        losses, _  = train(model, tr_dataloader, criterion, optimizer)

        losses, _  = validation(model, val_dataloader, criterion)

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
