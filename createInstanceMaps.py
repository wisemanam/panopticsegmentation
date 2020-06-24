from skimage.measure import label, regionprops
import numpy as np
import torch


def get_centers(instance_centers, threshold=0.7):

    found_centers = False
    while not found_centers:
        # creates numpy array of each instance center greater than threshold
        thresholded = (instance_centers >= threshold).astype(np.float32)

        lbl = label(thresholded)  # labeled array, connected region assigned same int value
        blobs = regionprops(lbl)  # the properties of regions in labeled array

        
        centers = [blob.centroid for blob in blobs] # the centroid of each labeled region
        try:
            probs = [instance_centers[int(y), int(x)] for (y, x) in centers] # list of probabilities
        except:
            print(centers)
            exit()

        if len(centers) == 0: # if no centers were found
            threshold -= 0.1  # decrease the threshold
        else:
            found_centers = True

        # TODO do topk

    return centers, probs  # return centroid of each region and list of probabilities


def create_instance_maps(segmentation_map, instance_centers, instance_regressions):
    # segmentation_map is of shape (H, W)
    # instance_centers is of shape (H, W)
    # instance_regressions is of shape (2, H, W)

    instance_classes = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
    centers, probs = get_centers(instance_centers)

    x, y = instance_regressions[0], instance_regressions[1]
    
    x_coords = np.tile(np.expand_dims(np.arange(x.shape[1]), 0), (x.shape[0], 1))+1 
    y_coords = np.tile(np.expand_dims(np.arange(x.shape[0]), 1), (1, x.shape[1]))+1
    
    offset_x = x_coords - x # horizontal offset between pixel and center
    offset_y = y_coords - y # vertical offset between pixel and center
    
    h, w = segmentation_map.shape
    instance_maps = np.zeros((h, w))  # instance maps are the same size as segmentation maps filled with 0s
    unique_instances = []
    for y in range(h):
        for x in range(w):
            seg_class = segmentation_map[y, x] # class of each pixel
            if seg_class not in instance_classes:
                continue
            center_pred_y, center_pred_x = offset_y[y, x], offset_x[y, x] # uses offsets to locate the center
            
            closest_dist, closest_id, closest_prob = 10000000, -1, -1
            # finds closest center
            for i, (center_y, center_x) in enumerate(centers):
                dist = (center_y-center_pred_y)**2 + (center_x-center_pred_x)**2 
                if dist < closest_dist:
                    closest_dist = dist
                    closest_id = i+1
            instance_maps[y, x] = closest_id
            if closest_id not in unique_instances:
                unique_instances.append(closest_id)

    return instance_maps, unique_instances, probs  # Returns the instance map of shape (H, W), a list of unique instance ids, and a list of probabilities
    

def separate_instance_maps(segmentation_map, instance_maps, segmentation_probs, unique_instances, inst_probs):
    # segmentation_map has shape (H, W)
    # instance_maps has shape (H, W)
    # segmentation_probs has shape (C, H, W)
    # unique_instances is a list of integer instance ids
    # inst_probs is a list of probability scores

    #print(segmentation_map.shape)
    #print(instance_maps.shape)
    #print(segmentation_probs.shape)
    #print(unique_instances)
    #print(inst_probs)

    binary_maps, inst_classes, inst_existance_probs = [],  [],  []
    for i, unique_instance in enumerate(unique_instances):
        inst_binary_map = (instance_maps == unique_instance).astype(np.uint8)

        inst_seg_map = segmentation_map[instance_maps == unique_instance]

        unique_elements, counts_elements = np.unique(inst_seg_map, return_counts=True)
        hist = list(zip(unique_elements, counts_elements))
        hist = sorted(hist, key=lambda x: x[1])
        inst_class = hist[-1][0]

        inst_seg_probs = segmentation_probs[inst_class, instance_maps == unique_instance]
        inst_seg_prob = np.mean(inst_seg_probs)

        inst_fin_prob = inst_seg_prob*inst_probs[i-1]

        binary_maps.append(inst_binary_map)
        inst_classes.append(inst_class)
        inst_existance_probs.append(inst_fin_prob)

    return binary_maps, inst_classes, inst_existance_probs
