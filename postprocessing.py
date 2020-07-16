from torch import nn
import torch
import numpy as np
import torch.nn.functional as F

import matplotlib.pyplot as plt

class NMS(nn.Module):
    def __init__(self, kernel_size=7, threshold=0.1, top_k=200):
        super(NMS, self).__init__()

        self.top_k = top_k

        self.max_pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size//2)
        self.threshold = nn.Threshold(threshold, 0)

    def forward(self, center_map):
        # Center Map should be of shape (B, 1, H, W)
        # Returns a list of centers and probs [(c1, p1), ...] where c1 is of shape (K, 2) and p1 is of shape (K, )

        pooled_centers = self.max_pool(center_map)
        thresholded_centers = self.threshold(pooled_centers)

        center_map[thresholded_centers != center_map] = 0

        center_map = center_map.squeeze(1)

        all_centers = []
        for nms_map in center_map:
            nonzero_inds = torch.nonzero(nms_map, as_tuple=False)

            nonzero_probs = nms_map[nonzero_inds[:, 0], nonzero_inds[:, 1]]

            sorted_inds = torch.argsort(nonzero_probs, descending=True)
            top_k_inds = sorted_inds[:self.top_k]

            sorted_coords = nonzero_inds[top_k_inds]
            sorted_probs = nonzero_probs[top_k_inds]
            all_centers.append((sorted_coords, sorted_probs))

        return all_centers


class CreateInstMap(nn.Module):
    def __init__(self, dims=(512, 1024)):
        super(CreateInstMap, self).__init__()

        h, w = dims

        x_coords = np.tile(np.expand_dims(np.arange(w), 0), (h, 1)) + 1
        y_coords = np.tile(np.expand_dims(np.arange(h), 1), (1, w)) + 1

        x_coords = torch.from_numpy(x_coords)
        y_coords = torch.from_numpy(y_coords)

        xy_coords = torch.stack((x_coords, y_coords), 0)

        self.register_buffer('xy_coords', xy_coords)

    def forward(self, instance_regressions, center_coords):
        # segmentation_map should be of shape (H, W)
        # instance_regressions should be of shape (2, H, W)
        # center_coords should be of shape (K, 2)
        # Returns the instance map of shape (H, W)

        center_coords_pred = self.xy_coords - instance_regressions  # (2, H, W)
        center_coords_pred = center_coords_pred.unsqueeze(-1)  # (2, H, W, 1)

        k, _ = center_coords.shape

        y_centers, x_centers = center_coords[:, 0], center_coords[:, 1]

        center_batch_t = torch.stack((x_centers, y_centers), 0).view(2, 1, 1, k)  # (2, 1, 1, K)

        dist = center_coords_pred - center_batch_t  # (2, H, W, K)
        dist = torch.sqrt(torch.sum(dist * dist, 0))  # (H, W, K)

        closest_inds = torch.argmin(dist, -1)+1  # (H, W)

        return closest_inds


class SeperateInstMap(nn.Module):
    def __init__(self, dims=(512, 1024)):
        super(SeperateInstMap, self).__init__()

        h, w = dims

        self.threshold = nn.Threshold(23.99, 50)

        ones = torch.ones((h, w))
        zeros = torch.zeros((h, w))

        self.register_buffer('ones', ones)
        self.register_buffer('zeros', zeros)

    def forward(self, segmentation_map, instance_maps, seg_map_probs, instance_probs):
        # obtains all pixels corresponding to "thing" classes
        thresh_map = self.threshold(segmentation_map)
        things_segs = torch.where(thresh_map <= 33, self.ones, self.zeros)

        instance_maps = instance_maps*things_segs

        unique_insts = torch.unique(instance_maps)

        inst_maps = []  # (instance_map, instance_prob, segmentation_prob, n_pixels)

        for inst in unique_insts:
            if inst == 0:
                continue

            inst_prob = instance_probs[int(inst)-1]

            inst_map = instance_maps == inst

            seg_ids, seg_counts = torch.unique(segmentation_map[inst_map], return_counts=True)

            inst_class = int(seg_ids[torch.argsort(seg_counts)[-1]])

            seg_prob = seg_map_probs[inst_class][inst_map].mean()

            inst_map = inst_map.type(torch.uint8)

            inst_maps.append((inst_map, inst_class, float(inst_prob), float(seg_prob), int(seg_counts.sum())))

        return inst_maps


class PostProcessing(nn.Module):
    def __init__(self, dims=(512, 1024), kernel_size=7, top_k=200):
        super(PostProcessing, self).__init__()

        self.nms = NMS(kernel_size=kernel_size, top_k=top_k)
        self.create_inst_map = CreateInstMap(dims)
        self.sep_inst_map = SeperateInstMap(dims)

    def forward(self, segmentation_probs, center_maps, regression_maps, gt_seg=None):
        """

        :param segmentation_probs: Shape (B, C, H, W)
        :param center_maps: Shape (B, 1, H, W)
        :param regression_maps: (B, 2, H, W)
        :param gt_seg: The optional ground-truth segmentation map (B, H, W) - used to test the instance branch outputs
        :return: returns a list of lists containing the instance maps for each sample in the batch.
        It is in the format [[(binary_map, inst_class, inst_prob, seg_prob, n_pixels), ...], ...]
        """

        seg_probs = torch.softmax(segmentation_probs, 1)
        if gt_seg is None:
            segs = torch.argmax(segmentation_probs, 1)
        else:
            segs = gt_seg[:, 0]

        outs = self.nms(center_maps)

        outputs = []
        inst_maps = []

        for i in range(len(outs)):
            centers, probs = outs[i]
            if centers.shape[0] == 0:
                outputs.append([])
                continue

            inst_map = self.create_inst_map(regression_maps[i], centers)

            instance_maps = self.sep_inst_map(segs[i], inst_map, seg_probs[i], probs)

            outputs.append(instance_maps)
            inst_maps.append(inst_map)

        return outputs, segs, inst_maps


class PostProcessing2(nn.Module):
    def __init__(self, kernel_size=7, dims=(512, 1024), top_k=200):
        super(PostProcessing2, self).__init__()

        self.kernel_size = kernel_size*2
        self.top_k = top_k

        self.max_pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size//2)

        self.class_thresh = nn.Threshold(23.99, 50)

        self.h, self.w = dims

        h, w = dims

        ones = torch.ones((h, w))
        zeros = torch.zeros((h, w))

        self.register_buffer('ones', ones)
        self.register_buffer('zeros', zeros)

        x_coords = np.tile(np.expand_dims(np.arange(w), 0), (h, 1)) + 1
        y_coords = np.tile(np.expand_dims(np.arange(h), 1), (1, w)) + 1

        x_coords = torch.from_numpy(x_coords)
        y_coords = torch.from_numpy(y_coords)

        xy_coords = torch.stack((x_coords, y_coords), 0)

        self.register_buffer('xy_coords', xy_coords.unsqueeze(0))

        box_filter = torch.ones((1, 1, kernel_size, kernel_size))
        # box_filter = torch.ones((1, 1, 1, 1)) # If using ground truth segmentations

        self.register_buffer('box_filter', box_filter)

        self.thresh_center = nn.Threshold(50, 0)

    def get_vote_map(self, center_coords_pred, segmentation_map):

        things_segs = self.class_thresh(segmentation_map)
        things_segs = things_segs <= 33 # boolean tensor
        vote_map = torch.zeros_like(segmentation_map)

        for i, votes in enumerate(center_coords_pred):
            things_seg = things_segs[i, 0]

            votes = votes[:, things_seg].round().long()

            # if there are no 'thing' objects
            if votes.shape[1] == 0:
                continue

            # Ensures all votes fall within the image bounds
            sub_vals, _ = torch.min(votes, 0)
            votes = votes[:, sub_vals >= 0]
            votes = votes[:, votes[0] < self.w]
            votes = votes[:, votes[1] < self.h]

            if votes.shape[1] == 0:
                continue

            # gets all unique votes and counts
            unique_votes, unique_vote_counts = torch.unique(votes, dim=1, return_counts=True)

            # at each pixel, counts votes for said pixel
            vote_map[i, 0][unique_votes[1], unique_votes[0]] = unique_vote_counts.float()

        return vote_map, things_segs

    def get_centers(self, nms_map, aggr_vote_region_map):
        nonzero_inds = torch.nonzero(nms_map, as_tuple=False)

        nonzero_counts = aggr_vote_region_map[nonzero_inds[:, 0], nonzero_inds[:, 1]]

        if nonzero_inds.shape[0] == 0:
            return None, None

        sorted_inds = torch.argsort(nonzero_counts, descending=True)
        top_k_inds = sorted_inds[:self.top_k]

        sorted_coords = nonzero_inds[top_k_inds]
        sorted_counts = nonzero_counts[top_k_inds]

        return sorted_coords, sorted_counts

    def create_inst_maps(self, center_coords_pred, center_coords, things_segs):
        center_coords_pred = center_coords_pred.unsqueeze(-1)  # (2, H, W, 1)

        k, _ = center_coords.shape

        y_centers, x_centers = center_coords[:, 0], center_coords[:, 1]

        center_batch_t = torch.stack((x_centers, y_centers), 0).view(2, 1, 1, k)  # (2, 1, 1, K)

        dist = center_coords_pred - center_batch_t  # (2, H, W, K)
        dist = torch.sqrt(torch.sum(dist * dist, 0))  # (H, W, K)

        closest_inds = torch.argmin(dist, -1) + 1  # (H, W)

        instance_map = closest_inds * things_segs

        return instance_map

    def separate_inst_maps(self, instance_map, segmentation_map, seg_map_probs):
        unique_insts = torch.unique(instance_map)

        inst_maps = []  # (instance_map, class_ind, instance_prob, segmentation_prob, n_pixels)

        for inst in unique_insts:
            if inst == 0:
                continue

            inst_prob = 1.0  # instance_probs[int(inst) - 1]

            inst_map = instance_map == inst

            seg_ids, seg_counts = torch.unique(segmentation_map[inst_map], return_counts=True)

            inst_class = int(seg_ids[torch.argsort(seg_counts)[-1]])

            seg_prob = seg_map_probs[inst_class][inst_map].mean()

            inst_map = inst_map.type(torch.uint8)

            inst_maps.append((inst_map, inst_class, float(inst_prob), float(seg_prob), int(seg_counts.sum())))

        return inst_maps

    def remove_high_mean_centers(self, inst_map, regressions, threshold=1):
        # Removes instances with average offsets above a given threshold
        unique_instances = torch.unique(inst_map)
        for instance in unique_instances:
            regressions = regressions.squeeze()  # shape: (2, H, W) # regressions for all instances
            regressions1 = regressions[:, inst_map == instance]  # shape: (2, N) # regressions for instances
            if abs(torch.mean(regressions1, 1)[0].item()) > threshold or abs(torch.mean(regressions1, 1)[1].item()) > threshold:
                instance = 0
            if instance == 0:
                continue
            print(instance, 'num_pixels =', regressions1.shape[1], 'Mean Regressions:', torch.mean(regressions1, 1))
        return inst_map

    def b_box_ratio_pruning(self, inst_map, sorted_coords, threshold = 0.3):
        unique_instances = torch.unique(inst_map)
        for instance in unique_instances:
            if instance == 0:
                continue
            n_pixels = 0
            top, bottom, left, right = [self.h, 0, self.w, 0]
            for row in range(len(inst_map)):
                for column in range(len(inst_map[row])):
                    if inst_map[row][column] == instance:
                        n_pixels += 1
                        if row < top:
                            top = row
                        if row > bottom:
                            bottom = row
                        if column < left:
                            left = column
                        if column > right:
                            right = column
            b_box_ratio = n_pixels / ((bottom - top) * (right - left))
            
            if b_box_ratio < threshold:
                
                # Locates center of instance
                center_coord = [0, 0]
                for center in sorted_coords:
                    if inst_map[center[0]][center[1]] == instance:
                        center_coord = center

                # Finds next closest instance
                min_dist = 10000000
                new_instance = 0
                for center in sorted_coords:
                    dist = (center[0] - center_coord[0])**2 + (center[1] - center_coord[1])**2
                    if dist < min_dist and dist != 0:
                        min_dist = dist
                        new_instance = inst_map[center[0]][center[1]]

                # Sets all pixels in old instance to new instance
                for row in range(len(inst_map)):
                    for column in range(len(inst_map[row])):
                        if inst_map[row][column] == instance:
                            inst_map[row][column] = new_instance
                            
        return inst_map


    def forward(self, segmentation_probs, center_maps_placeholder, center_regressions, gt_seg=None):
        # center_regressions should be of shape (B, 2, H, W)
        # segmentation_map should be of shape (B, 1, H, W)
        # Returns a list of centers and probs [(c1, p1), ...] where c1 is of shape (K, 2) and p1 is of shape (K, )

        # seg_probs converts logits into probability using softmax
        seg_probs = torch.softmax(segmentation_probs, 1)
        if gt_seg is None:
            segmentation_map = torch.argmax(segmentation_probs, 1).unsqueeze(1).float()
        else:
            segmentation_map = gt_seg

        #converts offsets into votes
        center_coords_pred = self.xy_coords - center_regressions

        vote_map, things_segs = self.get_vote_map(center_coords_pred, segmentation_map)

        # performs operation to sum all nearby votes (since not all votes are exactly on the center)
        aggr_vote_region_map = F.conv2d(vote_map, self.box_filter, padding=self.kernel_size // 2, stride=1)

        # performs NMS on the vote map
        vote_map = aggr_vote_region_map
        pooled_centers = self.max_pool(vote_map)
        vote_map[pooled_centers != vote_map] = 0

        aggr_vote_region_map = aggr_vote_region_map.squeeze(1)

        center_map = vote_map.squeeze(1)

        center_map = self.thresh_center(center_map) # only counts centers with more votes than threshold

        outputs = []
        inst_maps = []
        for i, nms_map in enumerate(center_map):
            sorted_coords, sorted_counts = self.get_centers(nms_map, aggr_vote_region_map[i])
            # sorted_coords: coordinates of centers sorted by number of votes

            if sorted_coords is None:
                outputs.append([])
                continue

            inst_map = self.create_inst_maps(center_coords_pred[i], sorted_coords, things_segs[i, 0])

            # removes centers with high average offsets
            inst_map = self.remove_high_mean_centers(inst_map, center_regressions, threshold=1)

            # inst_map = self.b_box_ratio_pruning(inst_map, sorted_coords, threshold = 0.3)

            instance_maps = self.separate_inst_maps(inst_map, segmentation_map[i, 0], seg_probs[i])

            outputs.append(instance_maps)
            inst_maps.append(inst_map)

        return outputs, segmentation_map[:, 0], inst_maps
