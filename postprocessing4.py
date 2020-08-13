from torch import nn
import torch
# print(torch.__version__)
# exit()
import numpy as np
import torch.nn.functional as F

class PostProcessing4(nn.Module):
    def __init__(self, kernel_size=7, dims=(512, 1024), top_k=200, circle_radius=5):
        super(PostProcessing4, self).__init__()

        self.kernel_size = kernel_size
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

        kernel_size = 1
        box_filter = torch.ones((1, 1, kernel_size, kernel_size))

        self.register_buffer('box_filter', box_filter)

        self.thresh_center = nn.Threshold(10, 0)

        self.circle_radius = circle_radius
        k_size = self.circle_radius * 2 + 1
        self.circle_k_size = k_size

        x_coords = np.tile(np.expand_dims(np.arange(k_size), 0), (k_size, 1)) + 1
        y_coords = np.tile(np.expand_dims(np.arange(k_size), 1), (1, k_size)) + 1

        x_coords = torch.from_numpy(x_coords)
        y_coords = torch.from_numpy(y_coords)

        xy_coords = torch.stack((x_coords, y_coords), 0) - (k_size + 1) // 2
        xy_coords = xy_coords.float()

        self.register_buffer('circle_coords', xy_coords.view(1, 2, 1, 1, k_size, k_size))

        # circle_dists = xy_coords.square().sum(0, keepdim=True).sqrt()
        circle_dists = (xy_coords**2).sum(0, keepdim=True).sqrt()
        circle_mask = (circle_dists <= self.circle_radius).float()
        n_pixels_in_circle = circle_mask.sum()

        self.register_buffer('circle_mask', circle_mask.view(1, 1, 1, 1, k_size, k_size))
        self.register_buffer('n_pixels_in_circle', n_pixels_in_circle)

        self.register_buffer('center_threshold', (circle_dists*circle_mask).sum()/n_pixels_in_circle)

    def get_vote_map(self, center_coords_pred, segmentation_map):

        things_segs = self.class_thresh(segmentation_map)
        things_segs = things_segs <= 33

        vote_map = torch.zeros_like(segmentation_map)

        for i, votes in enumerate(center_coords_pred):
            things_seg = things_segs[i, 0]

            votes = votes[:, things_seg].round().long()
            if votes.shape[1] == 0:
                continue

            # Ensures all votes fall within the image bounds
            sub_vals, _ = torch.min(votes, 0)

            votes = votes[:, sub_vals >= 0]
            votes = votes[:, votes[0] < self.w]
            votes = votes[:, votes[1] < self.h]

            if votes.shape[1] == 0:
                continue

            unique_votes, unique_vote_counts = torch.unique(votes, dim=1, return_counts=True)

            vote_map[i, 0][unique_votes[1], unique_votes[0]] = unique_vote_counts.float()

        return vote_map, things_segs

    def get_centers(self, nms_map, aggr_vote_region_map):
        nonzero_inds = torch.nonzero(nms_map, as_tuple=False)  # (N, 2)

        if nonzero_inds.shape[0] == 0:
            return None, None

        nonzero_counts = aggr_vote_region_map[nonzero_inds[:, 0], nonzero_inds[:, 1]]

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

            inst_prob = 1.0#instance_probs[int(inst) - 1]

            inst_map = instance_map == inst

            seg_ids, seg_counts = torch.unique(segmentation_map[inst_map], return_counts=True)

            inst_class = int(seg_ids[torch.argsort(seg_counts)[-1]])

            seg_prob = seg_map_probs[inst_class][inst_map].mean()

            inst_map = inst_map.type(torch.uint8)

            inst_maps.append((inst_map, inst_class, float(inst_prob), float(seg_prob), int(seg_counts.sum())))

        return inst_maps

    def forward(self, segmentation_logits, center_maps_placeholder, center_regressions, gt_seg=None):
        # center_regressions should be of shape (B, 2, H, W)
        # segmentation_map should be of shape (B, 1, H, W)
        # Returns a list of centers and probs [(c1, p1), ...] where c1 is of shape (K, 2) and p1 is of shape (K, )

        seg_probs = torch.softmax(segmentation_logits, 1)
        if gt_seg is None:
            segmentation_map = torch.argmax(segmentation_logits, 1).unsqueeze(1).float()
        else:
            segmentation_map = gt_seg

        center_coords_pred = self.xy_coords - center_regressions

        things_segs = self.class_thresh(segmentation_map)
        things_segs = things_segs <= 33

        # Computes center maps based on some circular region around each pixel
        regression_patches = get_patches(center_regressions, self.circle_k_size)

        patches_offset = self.circle_coords - regression_patches
        # patches_offset_dist = patches_offset.square().sum(1, keepdim=True).sqrt()
        patches_offset_dist = (patches_offset**2).sum(1, keepdim=True).sqrt()

        patches_offset_dist = (patches_offset_dist * self.circle_mask).sum((-1, -2)) / self.n_pixels_in_circle

        vote_map = 0 - patches_offset_dist - 1
        aggr_vote_region_map = vote_map.squeeze(1)

        # performs NMS on the vote map
        pooled_centers = self.max_pool(vote_map)
        vote_map[pooled_centers != vote_map] = 0

        vote_map[vote_map <= (-self.center_threshold-1)] = 0

        center_map = vote_map.squeeze(1)

        outputs = []
        inst_maps = []
        for i, nms_map in enumerate(center_map):
            sorted_coords, sorted_counts = self.get_centers(nms_map, aggr_vote_region_map[i])

            if sorted_coords is None:
                outputs.append([])
                continue

            inst_map = self.create_inst_maps(center_coords_pred[i], sorted_coords, things_segs[i, 0])

            instance_maps = self.separate_inst_maps(inst_map, segmentation_map[i, 0], seg_probs[i])

            outputs.append(instance_maps)
            inst_maps.append(inst_map)

        return outputs, segmentation_map[:, 0], inst_maps


def get_patches(regressions, kernel_size=3):
    regressions = F.pad(regressions, [kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2])

    b, _, h, w = regressions.shape

    idxs_h = [[(h_idx + k_idx) for h_idx in range(0, h - kernel_size + 1)] for k_idx in range(0, kernel_size)]
    idxs_w = [[(w_idx + k_idx) for w_idx in range(0, w - kernel_size + 1)] for k_idx in range(0, kernel_size)]

    x = regressions[:, :, idxs_h, :]
    x = x[:, :, :, :, idxs_w]
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()

    return x
