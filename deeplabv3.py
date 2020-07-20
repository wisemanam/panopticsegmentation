import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import os

from resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
from aspp import ASPP, ASPP_Bottleneck
from decoder import seg_decoder, inst_decoder
from capsules import PrimaryCaps, ConvCaps, CapsulePooling


class Model(nn.Module):
    def __init__(self, model_id, project_dir):
        super(Model, self).__init__()
        self.num_classes = config.n_classes
        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        self.resnet = ResNet50_OS16()  # NOTE! specify the type of ResNet here

        self.aspp_seg = ASPP_Bottleneck()
        self.aspp_inst = ASPP_Bottleneck()

        in_feats = 1280
        self.segmentation_decoder = seg_decoder(in_feats=in_feats, num_classes=self.num_classes)
        self.instance_decoder = inst_decoder(in_feats=in_feats)

    def forward(self, x, gt_seg=None):
        # (x has shape (batch_size, 3, h, w))
        h = x.size()[2]
        w = x.size()[3]

        # Encoder:
        feature_map, skip_8, skip_4 = self.resnet(x)  # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNe$

        # Decoder for semantic segmentation:
        output = self.aspp_seg(feature_map)  # (shape: (batch_size, num_classes, h/16, w/16))
        output = self.segmentation_decoder(output, skip_8, skip_4)

        # Decoder for instance segmentation:
        features = self.aspp_inst(feature_map)
        center, regressions = self.instance_decoder(features, skip_8, skip_4)
        center = F.sigmoid(center)
        regressions = F.tanh(regressions)

        output = F.upsample(output, size=(h, w), mode="bilinear")  # (shape: (batch_size, num_classes, h, w))
        center = F.upsample(center, size=(h, w), mode="bilinear")
        regressions = F.upsample(regressions, size=(h, w), mode="bilinear")
        regressions[:, 0] = regressions[:, 0] * w
        regressions[:, 1] = regressions[:, 1] * h

        # Should output center with shape (B, 1, H/16, W/16)
        # and regressions with shape(B, 2, H/16, W/16)
        return output, center, regressions

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)

class Model2(nn.Module):
    def __init__(self, model_id, project_dir):
        super(Model2, self).__init__()
        self.num_classes = config.n_classes
        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        self.resnet = ResNet50_OS16()  # NOTE! specify the type of ResNet here

        self.aspp = ASPP_Bottleneck()

        in_feats = 16*4*4
            
        self.primary_caps = PrimaryCaps(in_feats, 32, (1, 1))
        self.caps_pooling = CapsulePooling((3,3), (1, 1), (1, 1))

        self.class_capsules = ConvCaps(32, 16, (1,1), padding=None)
            
        self.segmentation_decoder = seg_decoder(in_feats=in_feats, num_classes=self.num_classes)
        self.instance_decoder = inst_decoder(in_feats=in_feats)

    def forward(self, x, gt_seg=None):
        # (x has shape (batch_size, 3, h, w))
        h = x.size()[2]
        w = x.size()[3]

        # Encoder:
        feature_map, skip_8, skip_4 = self.resnet(x)  # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNe$
    
        # Capsules
        output = self.aspp(feature_map)  # (shape: (batch_size, 256, h/16, w/16))

        primary_capsules = self.primary_caps(output)  # (batch_size, h/16, w/16, 32*(4*4+1))

        primary_capsules_pooled = self.caps_pooling(primary_capsules)

        class_capsules = self.class_capsules(primary_capsules_pooled)  # (batch_size, h/16, w/16, C*(4*4+1))

        b, h_down, w_down, _ = class_capsules.shape
        c = 16
        p = 4

        poses, activations = class_capsules[..., :c*p*p], class_capsules[..., c*p*p:]  # Shapes (batch_size, h/16, w/16, C*4*4) and (batch_size, h/16, w/16, C)
        poses = poses.permute(0, 3, 1, 2).contiguous()

        # Decoder for semantic segmentation:
        output = self.segmentation_decoder(poses, skip_8, skip_4)

        # Decoder for instance segmentation:
        center, regressions = self.instance_decoder(poses, skip_8, skip_4)
        center = F.sigmoid(center)
        regressions = F.tanh(regressions)

        output = F.upsample(output, size=(h, w), mode="bilinear")  # (shape: (batch_size, num_classes, h, w))
        center = F.upsample(center, size=(h, w), mode="bilinear")
        regressions = F.upsample(regressions, size=(h, w), mode="bilinear")
        regressions[:, 0] = regressions[:, 0] * w
        regressions[:, 1] = regressions[:, 1] * h

        # Should output center with shape (B, 1, H/16, W/16)
        # and regressions with shape(B, 2, H/16, W/16)
        return output, center, regressions

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)


class CapsuleModel(nn.Module):
    def __init__(self, model_id, project_dir):
        super(CapsuleModel, self).__init__()
        self.num_classes = config.n_classes
        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        self.resnet = ResNet50_OS16()  # NOTE! specify the type of ResNet here

        self.aspp_seg = ASPP_Bottleneck()
        self.aspp_inst = ASPP_Bottleneck()

        in_feats = 1280

        self.primary_caps = PrimaryCaps(in_feats, 32, (1, 1))
        self.caps_pooling = CapsulePooling((3,3), (1, 1), (1, 1))

        self.class_capsules = ConvCaps(32, self.num_classes, (1,1), padding=None)

        self.conv_1x1 = nn.Conv2d(self.num_classes*4*4, 128, kernel_size=1)

        self.conv_1x1_c = nn.Conv2d(128, 1, kernel_size=1)
        self.conv_1x1_r = nn.Conv2d(128, 2, kernel_size=1)

    def forward(self, x, gt_seg=None):
        # (x has shape (batch_size, 3, h, w)), gt_seg has shape (batch_size, 1, h, w)
        h = x.size()[2]
        w = x.size()[3]

        # Encoder:
        feature_map, skip_8, skip_4 = self.resnet(x)  # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNe$

        # Decoder for semantic segmentation:
        output = self.aspp_seg(feature_map)  # (shape: (batch_size, 256, h/16, w/16))

        primary_capsules = self.primary_caps(output)  # (batch_size, h/16, w/16, 32*(4*4+1))

        primary_capsules_pooled = self.caps_pooling(primary_capsules)

        class_capsules = self.class_capsules(primary_capsules_pooled)  # (batch_size, h/16, w/16, C*(4*4+1))

        b, h_down, w_down, _ = class_capsules.shape
        c = self.num_classes
        p = 4

        poses, activations = class_capsules[..., :c*p*p], class_capsules[..., c*p*p:]  # Shapes (batch_size, h/16, w/16, C*4*4) and (batch_size, h/16, w/16, C)

        output = activations.permute(0, 3, 1, 2).contiguous()

        poses = poses.view(b, h_down, w_down, c, p*p)  # (b, h/16, h/16, C, 16)

        if gt_seg is None:
            mask_inds = torch.argmax(activations, -1)
        else:
            gt_seg[gt_seg == 255] = 0
            mask_inds = F.upsample(gt_seg.float(), size=(h_down, w_down), mode="nearest").squeeze(1)

        mask = F.one_hot(mask_inds.long(), self.num_classes).unsqueeze(-1).float()  # (b, h/16, h/16, C, 1)

        masked_poses = mask*poses

        masked_poses = masked_poses.view(b, h_down, w_down, c*p*p).permute(0, 3, 1, 2).contiguous()   # (b, C*P*P, h/16, h/16)

        decoder_1 = F.relu(self.conv_1x1(masked_poses))

        center = self.conv_1x1_c(decoder_1)
        regressions = self.conv_1x1_r(decoder_1)

        center = F.sigmoid(center)
        regressions = F.tanh(regressions)

        output = F.upsample(output, size=(h, w), mode="bilinear")  # (shape: (batch_size, num_classes, h, w))
        center = F.upsample(center, size=(h, w), mode="bilinear")
        regressions = F.upsample(regressions, size=(h, w), mode="bilinear")
        regressions[:, 0] = regressions[:, 0] * w
        regressions[:, 1] = regressions[:, 1] * h

        return output, center, regressions

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

