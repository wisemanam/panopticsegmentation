import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision import transforms
from PIL import Image
import config


def get_cityscapes_dataset(root='./CityscapesData/', train=True):
    split = 'train' if train else 'val'
    return CustomCityscapes(root, split=split, mode='fine', target_type=['semantic', 'instance'])

def custom_collate(batch):
    image = []
    segmentation_maps = []
    instance_centers= []
    instance_regressions = []
    instance_present = []
    segmentation_weights = []
    class_list = []
    point_list = []
    image_name = []
    for image1, (segmentation_maps1, instance_centers1, instance_regressions1, instance_present1, segmentation_weights1), class_list1, point_list1, image_name1 in batch:
       image.append(image1)
       segmentation_maps.append(torch.tensor(segmentation_maps1))
       instance_centers.append(torch.tensor(instance_centers1))
       instance_regressions.append(torch.tensor(instance_regressions1))
       instance_present.append(torch.tensor(instance_present1))
       segmentation_weights.append(torch.tensor(segmentation_weights1))
       class_list.append(torch.tensor(class_list1))
       for i in range(config.batch_size):
           point_list.append(torch.tensor(point_list1[i]))
       image_name.append(image_name1)
    image = torch.stack(image)
    segmentation_maps = torch.stack(segmentation_maps)
    instance_centers = torch.stack(instance_centers)
    instance_regressions = torch.stack(instance_regressions)
    instance_present = torch.stack(instance_present)
    segmentation_weights = torch.stack(segmentation_weights)
    # image_name = torch.stack(image_name)

    return image, (segmentation_maps, instance_centers, instance_regressions, instance_present, segmentation_weights), class_list, point_list, image_name



class CustomCityscapes(Cityscapes):
    def __init__(self, root, split, mode, target_type):
        super(CustomCityscapes, self).__init__(root, split=split, mode=mode, target_type=target_type)
        self.to_tensor = transforms.ToTensor()

        self.split = split

        self.gaussian = np.zeros((33, 33))
        for i in range(33):
            for j in range(33):
                self.gaussian[i, j] = 1/(2 * np.pi * 8 * 8) * np.exp(0 - ((i - 17) ** 2 + (j - 17) ** 2) / (2 * 8 * 8))

        self.gaussian = self.gaussian / self.gaussian.max()

        self._TRAIN_ID_TO_EVAL_ID = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

        self._EVAL_ID_TO_TRAIN_ID = [255, 255, 255, 255, 255, 255, 255, 0, 1, 255, 255, 2, 3, 4, 255, 255,
                                     255, 5, 255, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 255, 255, 16, 17, 18, 255]

        self.conv_eval_to_train = np.vectorize(lambda x: self._EVAL_ID_TO_TRAIN_ID[x])
        self.conv_train_to_eval = np.vectorize(lambda x: self._TRAIN_ID_TO_EVAL_ID[x])

    def __getitem__(self, index):
        img_name = self.images[index]
        h = config.h
        w = config.w

        image, (segmentation_maps, instance_maps) = super().__getitem__(index)

        if self.split == 'train':
            if np.random.random_sample([0, 1]) >= 0.5:
                image = transforms.functional.hflip(image)
                segmentation_maps = transforms.functional.hflip(segmentation_maps)
                instance_maps = transforms.functional.hflip(instance_maps)

            crop_perc = np.random.choice([x / 10 for x in range(5, 15)])
            image_w, image_h = image.size
            crop_w, crop_h = int(image_w * crop_perc), int(image_h * crop_perc)

            if crop_perc > 1:
                diff_w, diff_h = crop_w - image_w, crop_h - image_h
                left_padding = np.random.randint(0, diff_w)
                top_padding = np.random.randint(0, diff_h)
                pad_tuple = (left_padding, top_padding, crop_w - left_padding, crop_h - top_padding)

                image = transforms.functional.pad(image, pad_tuple, fill=255)
                segmentation_maps = transforms.functional.pad(segmentation_maps, pad_tuple, fill=255)
                instance_maps = transforms.functional.pad(instance_maps, pad_tuple, fill=255)

                image_w, image_h = image.size

            start_x = np.random.randint(0, image_w - crop_w + 1)
            start_y = np.random.randint(0, image_h - crop_h + 1)
            crop_tuple = (start_x, start_y, start_x + crop_w, start_y + crop_h)
            image = image.crop(crop_tuple)
            segmentation_maps = segmentation_maps.crop(crop_tuple)
            instance_maps = instance_maps.crop(crop_tuple)

        image = image.resize(size=(w, h), resample=Image.BILINEAR)
        segmentation_maps = segmentation_maps.resize(size=(w, h), resample=Image.NEAREST)
        instance_maps = instance_maps.resize(size=(w, h), resample=Image.NEAREST)

        instance_maps = np.array(instance_maps)
        center_map = np.zeros((h, w))
        instance_regressions = np.zeros((2, h, w))
        regression_present = np.zeros((h, w))
        segmentation_weights = np.ones((h, w))

        unique_values = np.unique(instance_maps)

        instance_values = [x for x in unique_values if x >= 1000]

        point_list = []
        class_list = []
        for instance in instance_values:
            pixels = np.stack(np.where(instance_maps == instance))
            
            point_list.append(pixels)

            center = np.round(np.mean(pixels, 1)).astype(np.int32)  # gives the center (y, x)
            center = np.array((min(h - 17, max(17, center[0])), min(w - 17, max(17, center[1]))))

            y, x = center[0], center[1]
            
            class_list.append(np.array(segmentation_maps)[pixels[0][0]][pixels[1][0]])

            center_map[y - 16: y + 17, x - 16: x + 17] = np.maximum(self.gaussian, center_map[y - 16: y + 17, x - 16: x + 17])

            dists = pixels - np.expand_dims(center, 1)

            instance_regressions[:, pixels[0], pixels[1]] = dists
            regression_present[pixels[0], pixels[1]] = 1

            if pixels.shape[1] <= 64 * 64:
                segmentation_weights[pixels[0], pixels[1]] = 3

        instance_regressions = np.concatenate((instance_regressions[1:], instance_regressions[:1]), 0)  # Changes from y-x to x-y

        instance_centers = np.expand_dims(center_map, 0)
        instance_present = np.expand_dims(regression_present, 0)
        segmentation_weights = np.expand_dims(segmentation_weights, 0)

        image = self.to_tensor(image)

        segmentation_maps = np.expand_dims(np.array(segmentation_maps), 0)  # (H, W)

        if config.n_classes == 19:
            segmentation_maps[segmentation_maps == 255] = 0
            segmentation_maps = self.conv_eval_to_train(segmentation_maps)
        elif config.n_classes == 34:
            segmentation_maps = segmentation_maps
        else:
            assert NotImplementedError, "Must have either 19 or 34 classes for Cityscapes"

        return image, (segmentation_maps, instance_centers, instance_regressions, instance_present, segmentation_weights), class_list, point_list, img_name 
