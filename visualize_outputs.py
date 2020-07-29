import numpy as np
import matplotlib.pyplot as plt
from vis_flow import flow_to_color
from torch import nn
import torch
import torch.nn.functional as F
import math
from PIL import Image
from postprocessing import PostProcessing, PostProcessing2
from postprocessing3 import PostProcessing3


def display_segmentations(gt_seg, pr_seg):
    fig = plt.figure(figsize=(2, 2))

    fig.add_subplot(2, 1, 1)

    plt.imshow(gt_seg, vmin=0, vmax=34)

    fig.add_subplot(2, 1, 2)

    plt.imshow(pr_seg, vmin=0, vmax=34)

    plt.show()


def display_centers(gt_center, pr_center):
    fig = plt.figure(figsize=(2, 2))

    fig.add_subplot(2, 1, 1)

    plt.imshow(gt_center)

    fig.add_subplot(2, 1, 2)

    plt.imshow(pr_center)

    plt.show()


def display_regressions(gt_reg, pr_reg):
    gt_reg_rgb = flow_to_color(gt_reg.transpose((1, 2, 0)))
    pr_reg_rgb = flow_to_color(pr_reg.transpose((1, 2, 0)))

    fig = plt.figure(figsize=(2, 2))

    fig.add_subplot(2, 1, 1)

    plt.imshow(gt_reg_rgb)

    fig.add_subplot(2, 1, 2)

    plt.imshow(pr_reg_rgb)

    plt.show()


def display_qualitative(seg, center, regr, inst):
    fig = plt.figure(figsize=(2, 2))

    fig.add_subplot(2, 2, 1)

    plt.imshow(seg, vmin=0, vmax=34)

    fig.add_subplot(2, 2, 2)

    plt.imshow(center*255)

    fig.add_subplot(2, 2, 3)

    gt_reg_rgb = flow_to_color(regr.transpose((1, 2, 0)))
    plt.imshow(gt_reg_rgb)

    fig.add_subplot(2, 2, 4)

    plt.imshow(inst)

    plt.show()


def main():
    vid_name = './DumpedOutputs/frankfurt/frankfurt_000000_001016'
    # vid_name = './DumpedOutputs/frankfurt/frankfurt_000000_000576'
    # vid_name = './DumpedOutputs/frankfurt/frankfurt_000001_007973'
    # vid_name = './DumpedOutputs/frankfurt/frankfurt_000000_013382'
    # vid_name = './DumpedOutputs/frankfurt/frankfurt_000000_001751'
    # vid_name = './DumpedOutputs/frankfurt/frankfurt_000001_046126'
    # vid_name = './DumpedOutputs/frankfurt/frankfurt_000000_003025'
    # vid_name = './DumpedOutputs/frankfurt/frankfurt_000000_005543'
    # vid_name = './DumpedOutputs/frankfurt/frankfurt_000000_008451'
    #vid_name = './DumpedOutputs/frankfurt/frankfurt_000000_005898'
    # vid_name = './DumpedOutputs/frankfurt/frankfurt_000000_011810'
    #vid_name = './DumpedOutputs/frankfurt/frankfurt_000000_012121'
    # vid_name = './DumpedOutputs/frankfurt/frankfurt_000000_013240'
    # vid_name = './DumpedOutputs/frankfurt/frankfurt_000000_013942'
    # vid_name = './DumpedOutputs/frankfurt/frankfurt_000000_015389'

    gt_seg = np.load(vid_name + '_gt_seg.npy')[0]
    gt_center = np.load(vid_name + '_gt_center.npy')[0]
    gt_regression = np.load(vid_name + '_gt_regression.npy')

    pr_seg = np.load(vid_name + '_pr_seg.npy')
    pr_center = np.load(vid_name + '_pr_center.npy')[0]
    pr_regression = np.load(vid_name + '_pr_regression.npy')

    # display_segmentations(gt_seg, np.argmax(pr_seg, 0))
    #
    # display_centers(gt_center, pr_center)
    #
    # display_regressions(gt_regression, pr_regression)

    postprocessing = PostProcessing2(dims=(512, 1024), kernel_size=7, top_k=200)

    with torch.no_grad():
        seg = torch.from_numpy(gt_seg).unsqueeze(0).unsqueeze(0)
        center = torch.from_numpy(gt_center).unsqueeze(0).unsqueeze(0)
        regression = torch.from_numpy(gt_regression).unsqueeze(0)

        seg_probs = torch.from_numpy(pr_seg).unsqueeze(0)
        #center = torch.from_numpy(pr_center).unsqueeze(0).unsqueeze(0)
        #regression = torch.from_numpy(pr_regression).unsqueeze(0)

        #outputs1, out_seg1, inst_maps1 = postprocessing(seg_probs, center, regression)
        outputs1, out_seg1, inst_maps1 = postprocessing(seg.float(), center, regression)

    plt.imshow(inst_maps1[0].numpy())
    plt.show()

if __name__ == '__main__':
    main()
