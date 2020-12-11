import numpy as np
from pycocotools import mask
from PIL import Image, ImagePalette # For indexed images

def cocoSegmentationToSegmentationMap(coco, target, img_size, checkUniquePixelLabel=True, includeCrowd=False):
    '''
    Convert COCO GT or results for a single image to a segmentation map.
    :param coco: an instance of the COCO API (ground-truth or result)
    :param imgId: the id of the COCO image
    :param checkUniquePixelLabel: (optional) whether every pixel can have at most one label
    :param includeCrowd: whether to include 'crowd' thing annotations as 'other' (or void)
    :return: labelMap - [h x w] segmentation map that indicates the label of each pixel
    '''

    # Init
    imgAnnots = target

    # Combine all annotations of this image in labelMap
    #labelMasks = mask.decode([a['segmentation'] for a in imgAnnots])
    labelMap = np.zeros(img_size, dtype=np.int32)
    instanceMap = np.zeros(img_size, dtype=np.int32)
    instance_count = np.zeros((92,), dtype=np.int32)
    for a in range(0, len(imgAnnots)):
        labelMask = coco.annToMask(imgAnnots[a]) == 1
        #labelMask = labelMasks[:, :, a] == 1
        newLabel = imgAnnots[a]['category_id']

        if checkUniquePixelLabel and (labelMap[labelMask] != 0).any():
            raise Exception('Error: Some pixels have more than one label (image %d)!' % (1))

        labelMap[labelMask] = newLabel

        instanceMap[labelMask] = 1000*newLabel + instance_count[newLabel]
        instance_count[newLabel] += 1


    return labelMap, instanceMap

