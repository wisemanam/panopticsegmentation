import PIL
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision import transforms
from PIL import Image
from math import pi, exp

def get_cityscapes_dataset(root='./CityscapesData/', train=True, download=True):
    transform = transforms.Compose(
         [transforms.Resize(size=(512,1024), interpolation=Image.BILINEAR), transforms.ToTensor()])
        # used to transform PIL image to pytorch tensor
    target_transform = transforms.Compose(
         [transforms.Resize(size=(512,1024), interpolation=Image.NEAREST), transforms.ToTensor()])

    if train:
        return CustomCityscapes(root, split='train', mode='fine', transform=transform, target_transform=target_transform, target_type='semantic')
    else:
        return CustomCityscapes(root, split='val', mode='fine', transform=transform, target_transform=target_transform, target_type='semantic')

class TrainDataset(Cityscapes):
    def __init__(self, Dataset):
        self.data = []  # this should hold a list of all samples
        self.images = []
        for i in Dataset:
            self.data.append(i)
            (x, y), name = i
            self.images.append(name)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        h = 1024
        w = 2048
        (x, y), image_name = self.data[index]
        (segmentation_maps, instance_maps), image = super().__getitem__(index)
        instance_centers, instance_regressions = np.zeros((1, h, w)), np.zeros((2, h, w))
        center = (0, 0)
        for instance in np.unique(instance_maps):
            if instance == 0:
                continue
            instance_centers, instance_regressions = np.zeros((1, h, w)), np.zeros((2, h, w))
            top, bottom, left, right = h, 0, w, 0
            for row in range(len(instance_maps[0])-1):
                if instance in instance_maps[0][row] and row < top:
                    top = row
                if instance in instance_maps[0][row] and row > bottom:
                    bottom = row
                for column in range(len(instance_maps[0][row])-1):
                    if instance == instance_maps[0][row][column] and column < left:
                        left = column
                    if instance == instance_maps[0][row][column] and column > right:
                        right = column
            instance_height = abs(bottom - top)
            instance_width = abs(right - left)
            center = (left + instance_width//2, top + instance_height//2)
            instance_centers[0, center[1], center[0]] = 1/(2*pi*(8**2)) * exp(-(center[1]**2 + center[0]**2)/(2*(8**2)))
        for row in range(len(instance_maps[0])-1):
            for column in range(len(instance_maps[0][row])-1):
                if instance_maps[0][row][column] == 0:  # if pixel is not part of an instance
                    continue
                else:
                    x_dist, y_dist = column - center[1], row - center[0]
                    instance_regressions[0][row][column], instance_regressions[1][row][column] = x_dist, y_dist
        return image, (segmentation_maps, instance_centers, instance_regressions), image_name

class ValidationDataset(Cityscapes):
    def __init__(self, Dataset):
        self.data = []  # this should hold a list of all samples
        for i in Dataset:
            self.data.append(i)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        h = 1024
        w = 2048
        (x, y), image_name = self.data[index]
        (segmentation_maps, instance_maps), image = super().__getitem__(index)
        instance_centers, instance_regressions = np.zeros((1, h, w)), np.zeros((2, h, w))
        center = (0, 0)
        for instance in np.unique(instance_maps):
            if instance == 0:
                continue
            instance_centers, instance_regressions = np.zeros((1, h, w)), np.zeros((2, h, w))
            top, bottom, left, right = h, 0, w, 0
            for row in range(len(instance_maps[0])-1):
                if instance in instance_maps[0][row] and row < top:
                    top = row
                if instance in instance_maps[0][row] and row > bottom:
                    bottom = row
                for column in range(len(instance_maps[0][row])-1):
                    if instance == instance_maps[0][row][column] and column < left:
                        left = column
                    if instance == instance_maps[0][row][column] and column > right:
                        right = column
            instance_height = abs(bottom - top)
            instance_width = abs(right - left)
            center = (left + instance_width//2, top + instance_height//2)
            instance_centers[0, center[1], center[0]] = 1/(2*pi*(8**2)) * exp(-(center[1]**2 + center[0]**2)/(2*(8**2)))
        for row in range(len(instance_maps[0])-1):
            for column in range(len(instance_maps[0][row])-1):
                if instance_maps[0][row][column] == 0:  # if pixel is not part of an instance
                    continue
                else:
                    x_dist, y_dist = column - center[1], row - center[0]
                    instance_regressions[0][row][column], instance_regressions[1][row][column] = x_dist, y_dist
        return image, (segmentation_maps, instance_centers, instance_regressions), image_name

class CustomCityscapes(Cityscapes):
    def __init__(self, root, split, mode, transform, target_transform, target_type):
        super(CustomCityscapes, self).__init__(root, split='train', mode='fine', transform=transform, target_transform=target_transform, target_type='semantic')

    def __getitem__(self, index):
        img_name = self.images[index]
        return super().__getitem__(index), img_name.split('leftImg8bit')[1]
