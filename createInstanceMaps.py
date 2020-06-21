from skimage.measure import label, regionprops

def get_centers(instance_centers, threshold=0.8):
    # creates numpy array of each instance center greater than threshold
    thresholded = (instance_centers >= threshold).astype(np.float32)
    
    
    lbl = label(thresholded) # labeled array, connected region assigned same int value
    blobs = regionprops(lbl) # the properties of regions in labeled array
    
    return [blob.centroid for blob in blobs] # return centroid of each region
    
def create_instance_maps(segmentation_map, instance_centers, instance_regressions):
    # segmentation_map is of shape (H, W)
    # instance_centers is of shape (H, W)
    # instance_regressions is of shape (2, H, W)

    instance_classes = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
    centers = get_centers(instance_centers)
    x, y = instance_regressions[0], instance_regressions[1] # distance from center
    
    x_coords = np.tile(np.expand_dims(np.arange(x.shape[1]), 0), (x.shape[0], 1))+1
    y_coords = np.tile(np.expand_dims(np.arange(x.shape[0]), 1), (1, x.shape[1]))+1
    
    offset_x = x_coords - x
    offset_y = y_coords - y
    
    h, w = segmentation_map.shape
    instance_maps = np.zeros((h, w)) # instance maps are the same size as segmentation maps filled with 0s
    for y in range(h):
        for x in range(w):
            seg_class = segmentation_map[y, x]
            if seg_class not in instance_classes:
                continue
            center_pred_y, center_pred_x = offset_y[y, x], offset_x[y, x]
            
            closest_dist, closest_id = 10000000, -1
            for i, (center_y, center_x) in enumerate(centers):
                dist = (center_y-center_pred_y)**2 + (center_x-center_pred_x)**2
                if dist < closest_dist:
                    closest_dist = dist
                    closest_id = i+1
            instance_maps[y, x] = closest_id
    return instance_maps # Returns the instance map of shape (H, W)
    
