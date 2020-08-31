import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import os

from resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
from aspp import ASPP, ASPP_Bottleneck
from decoder import seg_decoder, inst_decoder, by_regression_inst_decoder, seg_decoder2
from capsules import PrimaryCaps, ConvCaps, CapsulePooling
from HoughCapsules import HoughRouting1

class CapsuleModel3(nn.Module):
    def __init__(self, model_id, project_dir):
        super(CapsuleModel3, self).__init__()
        self.num_classes = config.n_classes
        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        self.resnet = ResNet50_OS16()  # NOTE! specify the type of ResNet here

        self.aspp = ASPP_Bottleneck()

        in_feats = 1280

        self.primary_caps = PrimaryCaps(in_feats, 32, (1, 1))
        self.caps_pooling = CapsulePooling((3, 3), (1, 1), (1, 1))

        self.class_capsules = ConvCaps(32, 16, (1, 1), padding=None)

        self.segmentation_decoder = seg_decoder2(in_feats=in_feats, num_classes=self.num_classes)
        self.instance_decoder = inst_decoder(in_feats=in_feats)

        self.linear = nn.Linear(in_feats, self.num_classes)

        self.hough_routing = HoughRouting1()

    def forward(self, x, point_lists=None, gt_seg=None):
        # (x has shape (batch_size, 3, h, w))
        h = x.size()[2]
        w = x.size()[3]

        # Encoder:
        feature_map, skip_8, skip_4 = self.resnet(x)  # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNe$

        # Capsules
        feature_output = self.aspp(feature_map)  # (shape: (batch_size, 256, h/16, w/16))

        class_capsules, poses = feature_output, feature_output  # we use the output of the aspp as our "capsules"

        # Decoder for semantic segmentation:
        output = self.segmentation_decoder(poses, skip_8, skip_4)
        output = F.sigmoid(output)

        # Decoder for instance segmentation:
        center, regressions = self.instance_decoder(poses, skip_8, skip_4)
        center = F.sigmoid(center)
        regressions = F.tanh(regressions)

        output = F.upsample(output, size=(h, w), mode="bilinear")  # (shape: (batch_size, num_classes, h, w))
        center = F.upsample(center, size=(h, w), mode="bilinear")
        regressions = F.upsample(regressions, size=(h, w), mode="bilinear")
        regressions[:, 0] = regressions[:, 0] * w
        regressions[:, 1] = regressions[:, 1] * h

        if point_lists is None:
            inst_maps, point_lists, segmentation_lists = self.hough_routing(output, regressions, gt_seg)
        else:
            inst_maps = []
            segmentation_lists = []

        class_outputs = []
        for i, point_list in enumerate(point_lists):

            class_outs = []
            for inst_points in point_list:
                # gather capsules corresponding to inst_points
                inst_points = torch.unique(inst_points // 16, dim=1)

                y_coords, x_coords = inst_points[0, :], inst_points[1, :]

                # perform routing on inst capsules to get class capsules
                inst_capsules = class_capsules[i, :, y_coords, x_coords]
                # print('inst_capsules:', inst_capsules.shape)
                inst_capsules = torch.transpose(inst_capsules, 0, 1)
                #print('inst_capsules:', inst_capsules.shape)
                linear_class_capsules = self.linear(inst_capsules)
                #print('linear_class_capsules:', linear_class_capsules.shape)
                linear_class_capsules = torch.mean(linear_class_capsules, 0)
                #print('linear_class_capsules:', linear_class_capsules.shape)

                # get activations from the class capsules
                class_output = F.softmax(linear_class_capsules, dim=-1)
                # print('class_output:', class_output.shape)

                class_outs.append(class_output)

            class_outputs.append(torch.stack(class_outs) if len(class_outs) != 0 else [])

        # Should output center with shape (B, 1, H/16, W/16)
        # and regressions with shape(B, 2, H/16, W/16)
        return output, center, regressions, class_outputs, inst_maps, segmentation_lists

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)
            
class CapsuleModel4(nn.Module):
    def __init__(self, model_id, project_dir):
        super(CapsuleModel4, self).__init__()
        self.num_classes = config.n_classes
        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        self.resnet = ResNet50_OS16()  # NOTE! specify the type of ResNet here

        self.aspp = ASPP_Bottleneck()

        in_feats = 1280

        self.primary_caps = PrimaryCaps(in_feats, 32, (1, 1))
        self.caps_pooling = CapsulePooling((3, 3), (1, 1), (1, 1))

        self.class_capsules = ConvCaps(32, 16, (1, 1), padding=None)

        self.segmentation_decoder = seg_decoder2(in_feats=in_feats, num_classes=self.num_classes)
        self.instance_decoder = inst_decoder(in_feats=in_feats)

        self.linear = nn.Linear(256, self.num_classes)

        self.hough_routing = HoughRouting1()
        
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        self.conv1x1_key = nn.Conv2d(in_feats, 256, 1)
        self.conv1x1_query = nn.Conv2d(in_feats, 256, 1)
        self.conv1x1_value = nn.Conv2d(in_feats, 256, 1)

    def forward(self, x, point_lists=None, gt_seg=None):
        # (x has shape (batch_size, 3, h, w))
        h = x.size()[2]
        w = x.size()[3]

        # Encoder:
        feature_map, skip_8, skip_4 = self.resnet(x)  # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNe$

        # Capsules
        feature_output = self.aspp(feature_map)  # (shape: (batch_size, 256, h/16, w/16))

        class_capsules, poses = feature_output, feature_output  # we use the output of the aspp as our "capsules"

        # Decoder for semantic segmentation:
        output = self.segmentation_decoder(poses, skip_8, skip_4)
        output = F.sigmoid(output)
        class_capsule_activations = output

        class_capsules_key = self.conv1x1_key(class_capsules)
        class_capsules_query = self.conv1x1_query(class_capsules)
        class_capsules_value = self.conv1x1_value(class_capsules)
        
        # Decoder for instance segmentation:
        center, regressions = self.instance_decoder(poses, skip_8, skip_4)
        center = F.sigmoid(center)
        regressions = F.tanh(regressions)

        output = F.upsample(output, size=(h, w), mode="bilinear")  # (shape: (batch_size, num_classes, h, w))
        center = F.upsample(center, size=(h, w), mode="bilinear")
        regressions = F.upsample(regressions, size=(h, w), mode="bilinear")
        regressions[:, 0] = regressions[:, 0] * w
        regressions[:, 1] = regressions[:, 1] * h

        if point_lists is None:
            inst_maps, point_lists, segmentation_lists = self.hough_routing(output, regressions, gt_seg)
        else:
            inst_maps = []
            segmentation_lists = []

        class_outputs = []
        for i, point_list in enumerate(point_lists):

            class_outs = []
            for inst_points in point_list:
                # gather capsules corresponding to inst_points
                inst_points = torch.unique(inst_points // 16, dim=1)

                y_coords, x_coords = inst_points[0, :], inst_points[1, :]

                inst_capsules_activations = class_capsule_activations[i, 0, y_coords, x_coords] #shape (p, )
                argsort_activations = torch.argsort(inst_capsules_activations, descending=True)[:1000000]

                
                y_coords_topk = y_coords[argsort_activations]
                x_coords_topk = x_coords[argsort_activations]
                

                instance_capsules_key  = class_capsules_key[i, :, y_coords_topk, x_coords_topk]
                # instance_capsules_query = class_capsules_query[i, :, y_coords_topk, x_coords_topk]
                # instance_capsules_value = class_capsules_value[i, :, y_coords_topk, x_coords_topk]
                
                instance_capsules_key = torch.transpose(instance_capsules_key, 0, 1).unsqueeze(1)
                # instance_capsules_query = torch.transpose(instance_capsules_query, 0, 1).unsqueeze(1)
                # instance_capsules_value = torch.transpose(instance_capsules_value, 0, 1).unsqueeze(1) # (p, 1, 256)
                
                # inst_capsules, _ = self.attention(instance_capsules_query, instance_capsules_key, instance_capsules_value) # (p, 1, 256)
                # inst_capsules = inst_capsules.squeeze(1) # (p, 256)
                #print(inst_capsules.shape)
                linear_class_capsules = self.linear(instance_capsules_key.squeeze(1))
                linear_class_capsules = torch.mean(linear_class_capsules, 0)

                # get activations from the class capsules
                class_output = F.softmax(linear_class_capsules, dim=-1)
                # print('class_output:', class_output.shape)

                class_outs.append(class_output)

            class_outputs.append(torch.stack(class_outs) if len(class_outs) != 0 else [])

        # Should output center with shape (B, 1, H/16, W/16)
        # and regressions with shape(B, 2, H/16, W/16)
        return output, center, regressions, class_outputs, inst_maps, segmentation_lists

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)

if __name__ == '__main__':
    print('Here')
