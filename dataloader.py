import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision import transforms
from PIL import Image
import os
from math import pi, exp

def get_cityscapes_dataset(root='./CityscapesData/', train=True, download=True):
    transform = transforms.Compose(
         [transforms.Resize(size=(512,1024), interpolation=Image.BILINEAR), transforms.ToTensor()])
        # used to transform PIL image to pytorch tensor
    # target_transform = transforms.Compose(
    #     [transforms.Resize(size=(512,1024), interpolation=Image.NEAREST), transforms.ToTensor()])

    if train:
	    return CustomCityscapes(root, split='train', mode='fine', transform=transform, target_type=['semantic', 'instance']
    else:
	    return CustomCityscapes(root, split='val', mode='fine', transform=transform, target_type=['semantic', 'instance']

class CustomCityscapes(Cityscapes):
    def __init__(self, root, split, mode, transform, target_type):
        super(CustomCityscapes, self).__init__(root, split='train', mode='fine', transform=transform$

    def __getitem__(self, index):
        img_name = self.images[index]
        h = 512
        w = 1024
        image, (segmentation_maps, instance_maps) = super().__getitem__(index)
        segmentation_maps = segmentation_maps.resize(size=(w, h), resample=Image.NEAREST)
        instance_maps = instance_maps.resize(size=(w//16, h//16), resample=Image.NEAREST)
        segmentation_maps = np.array(segmentation_maps)
        instance_maps = np.array(instance_maps)
        instance_centers, instance_regressions = np.zeros((1, h//16, w//16)), np.zeros((2, h//16, w/$
        center = (0, 0)
        for instance in np.unique(instance_maps):
            if instance == 0:
                continue
            top, bottom, left, right = h, 0, w, 0
            for row in range(len(instance_maps)-1):
                if instance in instance_maps[row] and row < top:
                    top = row
                if instance in instance_maps[row] and row > bottom:
                    bottom = row
                for column in range(len(instance_maps[row])-1):
                    if instance == instance_maps[row][column] and column < left:
                        left = column
                    if instance == instance_maps[row][column] and column > right:
                        right = column
            instance_height = abs(bottom - top)
            instance_width = abs(right - left)
            center = (right -  instance_width//2, top + instance_height//2)
            instance_centers[0, center[1]//16, center[0]//16] = float(1/(2*pi*(8**2)) * exp(-(center[1]**2 + center[0]**2)/(2*(8**2))))
        for row in range(len(instance_maps)-1):
            for column in range(len(instance_maps[row])-1):
                if instance_maps[row][column] == 0:  # if pixel is not part of an instance
                    continue
                else:
                    x_dist, y_dist = column - center[1], row - center[0]
                    instance_regressions[0][row//16][column//16], instance_regressions[1][row//16][column//16] = x_dist, y_dist

        segmentation_maps = torch.from_numpy(segmentation_maps)
        instance_maps = torch.from_numpy(instance_maps)
        return image, (segmentation_maps, instance_centers, instance_regressions), img_name

class TrainDataset(Cityscapes):
    def __init__(self, Dataset):
        self.data = []  # this should hold a list of all samples
        for i in Dataset:
           self.data.append(i)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]

        x, y = 0, 0  # placeholder for the sample input(s) and output(s)

        return x, y

class ValidationDataset(Cityscapes):
    def __init__(self, Dataset):
     self.data = []  # this should hold a list of all samples
        for i in Dataset:
            self.data.append(i)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]

        x, y = 0, 0  # placeholder for the sample input(s) and output(s)

        return {'x': x, 'y': y}

def get_cityscapes_dataset2(root='./CityscapesData/', train=True, download=True):
    transform = transforms.Compose(
         [transforms.Resize(size=(512,1024), interpolation=Image.BILINEAR), transforms.ToTensor()])
        # used to transform PIL image to pytorch tensor
    target_transform = transforms.Compose(
         [transforms.Resize(size=(512,1024), interpolation=Image.NEAREST), transforms.ToTensor()])

    if train:
	return TrainDataset(CustomCityscapes(root, split='train', mode='fine', transform=transform, target_transform=target_transform, target_type='semantic'))
    else:
	return TrainDataset(CustomCityscapes(root, split='val', mode='fine', transform=transform, target_transform=target_transform, target_type='semantic'))




