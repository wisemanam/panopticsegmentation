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
        # used to transform PIL Image to tensor

    if train:
	return CustomCityscapes(root, split='train', mode='fine', transform=transform, target_type=['semantic', 'instance'])
    else:
	return CustomCityscapes(root, split='val', mode='fine', transform=transform, target_type='semantic')

class CustomCityscapes(Cityscapes):
    def __init__(self, root, split, mode, transform, target_type):
        super(CustomCityscapes, self).__init__(root, split='train', mode='fine', transform=transform, target_type=['semantic', 'instance'])

        # used to transform segmentation map back to tensor
        self.seg_transform = transforms.ToTensor()

        # guassian for instance centers:
        self.gaussian = np.zeros((33,33))
        for i in range(33):
            for j in range(33):
                self.gaussian[i, j] = 1/(2*np.pi*8*8)* np.exp( 0-((i-17)**2 +(j-17)**2) / (2*8*8) )

    def __getitem__(self, index):
        img_name = self.images[index]
        h = 512
        w = 1024
        image, (segmentation_maps, instance_maps) = super().__getitem__(index)
        segmentation_maps = segmentation_maps.resize(size=(w, h), resample=Image.NEAREST)
        instance_maps = instance_maps.resize(size=(w, h), resample=Image.NEAREST)
        instance_maps = np.array(instance_maps)
        instance_centers, instance_regressions = np.zeros((1, h, w)), np.zeros((2, h, w))
        instance_present = np.zeros((1, h, w))

        centers = {} # holds the centers of each instance: centers[instance] = (x, y)
        b_box = {} # holds the bounding box of each instance:  b_box[instance] = [top, bottom, left, right]

        # finds the bounding box of each instance
        for row in range(len(instance_maps)):
            for column in range(len(instance_maps[row])):
                instance = instance_maps[row][column]
                if instance < 1000:
                    continue
                if instance not in b_box:
                    b_box[instance] = [h, 0, w, 0]
                top, bottom, left, right = b_box[instance]
                if row < top:
                    top = row
                if row > bottom:
                    bottom = row
                if column < left:
                    left = column
                if column > right:
                    right = column
                b_box[instance] = [top, bottom, left, right]

        # finds center for each instance in b_box
        for instance in b_box:
            if instance not in b_box:
                continue
            top, bottom, left, right = b_box[instance]
            instance_height = abs(bottom - top)
            instance_width = abs(right - left)
            center = (right -  instance_width//2, top + instance_height//2)
            center = (min(w - 17, max(17, center[0])), min(h-17, max(17, center[1])))
            centers[instance] = center
            instance_centers[0, center[1] - 16: center[1] + 17, center[0] - 16: center[0] + 17] += self.gaussian
        instance_centers = np.clip(instance_centers, 0, 1)

        # finds the distance between each pixel and the instance center
        for row in range(len(instance_maps)):
            for column in range(len(instance_maps[row])):
                if instance_maps[row][column] <  1000:  # if pixel is not part of an instance
                    continue
                else:
                    center = centers[instance_maps[row][column]]
                    x_dist, y_dist = column - center[0], row - center[1]
                    instance_regressions[0][row][column], instance_regressions[1][row][column] =  x_dist, y_dist
                    instance_present[0, row, column]

        segmentation_maps = self.seg_transform(segmentation_maps)*255
        instance_maps = torch.from_numpy(instance_maps)
        return image, (segmentation_maps, instance_centers, instance_regressions, instance_present), img_name




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
