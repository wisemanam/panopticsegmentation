import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


def get_centers(center_map, top_k=200):
    """
    
    :param center_map: A map of centers with shape (H, W). In this map, the lower the center (error) value the more
    likely it is to exist.
    :param top_k: The number of centers which will be obtained.
    :return: The top_k coordinates and their corresponding errors of shapes (N, 2) and (N, ) respectively
    """

    center_coords = torch.nonzero(center_map, as_tuple=False)  # Obtains all center points - Shape: (N, 2)

    if center_coords.shape[0] == 0:  # No centers are present
        return None, None

    center_errors = center_map[center_coords[:, 0], center_coords[:, 1]]  # Obtains the errors for the given points

    top_k_inds = torch.argsort(center_errors, descending=False)[:top_k]  # obtains the k indices of centers with least error

    sorted_center_coords = center_coords[top_k_inds]
    sorted_center_errors = center_errors[top_k_inds]

    return sorted_center_coords, sorted_center_errors


def create_inst_maps(center_coords_pred, center_coords, things_segs):
    """

    :param center_coords_pred: A map of center predictions with shape (2, H, W).
    :param center_coords: The list of center coordinates (N, 2)
    :param things_segs: A boolean map containing the foreground segmentation of shape (H, W).
    :return: Returns the instance map which maps all foreground pixels to the center which is closest to its prediction.
    """

    center_coords_pred = center_coords_pred.unsqueeze(-1)  # (2, H, W, 1)

    k, _ = center_coords.shape

    y_centers, x_centers = center_coords[:, 0], center_coords[:, 1]

    center_batch_t = torch.stack((x_centers, y_centers), 0).view(2, 1, 1, k)  # (2, 1, 1, K)

    dist = center_coords_pred - center_batch_t  # (2, H, W, K)
    dist = torch.sqrt(torch.sum(dist * dist, 0))  # (H, W, K)

    closest_inds = torch.argmin(dist, -1) + 1  # (H, W)

    instance_map = closest_inds.cuda() * things_segs.cuda()

    return instance_map


def get_instance_pixels(instance_map):
    """

    :param instance_map: An instance map of shape (H, W)
    :return: Returns the points and segmentations for each instance in the instance map. The points are of shape (2, N)
    and the segmentations are of shape (H, W).
    """

    point_list = []
    segmentation_list = []

    unique_insts = torch.unique(instance_map)

    for inst in unique_insts:
        if inst == 0:
            continue

        inst_map = (instance_map == inst)

        pixels = torch.stack(torch.where(inst_map), 0)

        inst_map = inst_map.type(torch.uint8)

        point_list.append(pixels)
        segmentation_list.append(inst_map)

    return point_list, segmentation_list


def get_patches(regressions, kernel_size=3):
    regressions = F.pad(regressions, [kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2])

    b, _, h, w = regressions.shape

    idxs_h = [[(h_idx + k_idx) for h_idx in range(0, h - kernel_size + 1)] for k_idx in range(0, kernel_size)]
    idxs_w = [[(w_idx + k_idx) for w_idx in range(0, w - kernel_size + 1)] for k_idx in range(0, kernel_size)]

    x = regressions[:, :, idxs_h, :]
    x = x[:, :, :, :, idxs_w]
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()

    return x


class HoughRouting1(nn.Module):
    def __init__(self, nms_kernel_size=7, dims=(512, 1024), top_k=200, circle_radius=5):
        super(HoughRouting1, self).__init__()

        self.kernel_size = nms_kernel_size
        self.top_k = top_k

        self.max_pool = nn.MaxPool2d(self.kernel_size, stride=1, padding=self.kernel_size//2)

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

        circle_dists = (xy_coords**2).sum(0, keepdim=True).sqrt()
        circle_mask = (circle_dists <= self.circle_radius).float()
        n_pixels_in_circle = circle_mask.sum()

        self.register_buffer('circle_mask', circle_mask.view(1, 1, 1, 1, k_size, k_size))
        self.register_buffer('n_pixels_in_circle', n_pixels_in_circle)

        self.register_buffer('center_threshold', (circle_dists*circle_mask).sum()/n_pixels_in_circle+0.5)

    def forward(self, fg_pred, center_regressions, gt_fg=None):
        # fg_pred should be of shape (B, 1, H, W)
        # center_regressions should be of shape (B, 2, H, W)
        # gt_fg should be of shape (B, 1, H, W)
        # Returns a list of instance maps, point lists, and segmentation maps

        center_coords_pred = self.xy_coords - center_regressions

        if gt_fg is None:
            things_segs = fg_pred >= 0.5
        else:
            things_segs = gt_fg >= 0.5

        # Computes center maps based on some circular region around each pixel

        regression_patches = get_patches(center_regressions, self.circle_k_size)  # Obtains the regressions around each pixel

        patches_offset = self.circle_coords - regression_patches  # converts regressions into center predictions

        patches_offset_error = (patches_offset**2).sum(1, keepdim=True).sqrt()  # calculates the magnitude of the center prediction errors

        patches_offset_error = (patches_offset_error * self.circle_mask).sum((-1, -2)) / self.n_pixels_in_circle  # averages the center prediction errors

        vote_map = patches_offset_error + 1

        pooled_centers = 0-self.max_pool(0-vote_map)  # performs NMS on the prediction error (lowest error is a center)
        vote_map[pooled_centers != vote_map] = 0
        vote_map[vote_map >= self.center_threshold] = 0  # Removes possible noisy centers (i.e. if all zeros regressions - the case in the gt regressions)

        center_maps = vote_map.squeeze(1)  # (B, H, W)

        outputs = []  # [(inst_map, point_list, segmentation_list)]
        for i, center_map in enumerate(center_maps):
            sorted_coords, sorted_counts = get_centers(center_map)

            if sorted_coords is None:
                outputs.append(([], [], []))
                continue

            inst_map = create_inst_maps(center_coords_pred[i], sorted_coords, things_segs[i, 0])

            point_list, segmentation_list = get_instance_pixels(inst_map)

            outputs.append((inst_map, point_list, segmentation_list))

        return zip(*outputs)
