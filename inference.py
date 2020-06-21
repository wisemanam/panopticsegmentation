import config
import numpy as np
import torch
from train import get_accuracy
from torchvision.utils import save_image
import os
from dataloader import DataLoader, ValidationDataset, get_cityscapes_dataset
from cityscapesScripts.cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling import main as eval_main

def inference(model, data_loader):
    model.eval()

    if config.use_cuda:
        model.cuda()

    y_list = []
    y_pred_list = []
    img_ind = 0


    for i, sample in enumerate(data_loader):
        (x, y), name = sample
        if config.use_cuda:
            x = x.cuda()
            y = y.cuda()

        y_pred = model(x)

        y_list.append(name)

        y_pred_softmax = torch.softmax(y_pred, 1)  # should give shape (B, H, W)

        for img in y_pred_softmax:
            img_name = './SavedImages/output_segmentation_%d.png' % img_ind
            save_image(img, img_name)
            img_ind += 1
            y_pred_list.append(img_name)

    eval_main(y_pred_list, y_list)

    print('Finished inference.')

if __name__ == '__main__':
    max_epoch = 1000000
    min_loss = 0
    for i in './SavedModels':
        model_info = i.split('_')
        epoch = model_info[1]
        loss = model_info[2].split('.')
        if float(loss) < min_loss:
            min_loss = loss
        if epoch > max_epoch:
            max_epoch = epoch
    model = Model2('Model2', 'SimpleSegmentation/')
    model.load_state_dict(torch.load(os.path.join(config.save_dir, 'model_{}_{:.4f}.pth'.format(max_epoch, min_loss)))['state_dict'])
    val_dataset = get_cityscapes_dataset('~/SimpleSegmentation/CityscapesData', False, download=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    print('Finished inference.')

