import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import os
from resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
from aspp import ASPP, ASPP_Bottleneck
from decoder import fg_decoder, inst_decoder
from HoughCapsules import HoughRouting1
from setTransformer import TransformerRouting


class CapsuleModelNew1(nn.Module):
    def __init__(self, model_id, project_dir):
        super(CapsuleModelNew1, self).__init__()
        self.num_classes = config.n_classes
        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        self.resnet = ResNet50_OS16()  # NOTE! specify the type of ResNet here

        self.aspp = ASPP_Bottleneck()

        in_feats = 1280

        self.fg_decoder = fg_decoder(in_feats=in_feats, num_classes=self.num_classes)
        self.instance_decoder = inst_decoder(in_feats=in_feats)

        self.hough_routing = HoughRouting1()

        self.conv1x1_poses = nn.Conv2d(in_feats, 256, 1)
        self.conv1x1_acts = nn.Conv2d(in_feats, 256, 1)

        self.noise_scale = 4.0

        if config.positional_encoding_type == 'addition':
            self.transformer_routing = TransformerRouting(n_feats_in=256, n_caps_out=self.num_classes, output_dim=16)
        elif config.positional_encoding_type == 'concat':
            self.transformer_routing = TransformerRouting(n_feats_in=258, n_caps_out=self.num_classes, output_dim=16)

        self.linear = nn.Linear(16, 1)

    def forward(self, x, point_lists=None, gt_seg=None):
        # (x has shape (batch_size, 3, h, w))
        h = x.size()[2]
        w = x.size()[3]

        # Encoder:
        feature_map, skip_8, skip_4 = self.resnet(
            x)  # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNe$

        feature_output = self.aspp(feature_map)  # (shape: (batch_size, 1280, h/16, w/16))

        # Decoder for semantic segmentation:
        fg_pred = self.fg_decoder(feature_output, skip_8, skip_4)
        fg_pred = F.sigmoid(fg_pred)

        # Decoder for instance segmentation:
        center, regressions = self.instance_decoder(feature_output, skip_8, skip_4)
        center = F.sigmoid(center)
        regressions = F.tanh(regressions)

        # Resizes the output maps
        fg_pred = F.upsample(fg_pred, size=(h, w), mode="bilinear")  # (shape: (batch_size, num_classes, h, w))
        center = F.upsample(center, size=(h, w), mode="bilinear")
        regressions = F.upsample(regressions, size=(h, w), mode="bilinear")
        regressions[:, 0] = regressions[:, 0] * w
        regressions[:, 1] = regressions[:, 1] * h

        # creates the first capsule layer
        capsule_poses = self.conv1x1_poses(feature_output)  # (batch_size, 256, h/16, w/16)
        capsule_acts_logits = self.conv1x1_acts(feature_output)  # (batch_size, 1, h/16, w/16)

        if self.training:
            capsule_acts_logits += ((torch.rand(
                *capsule_acts_logits.shape) - 0.5) * self.noise_scale).cuda()  # adds noise to force network to learn binary activations
        capsule_acts = F.sigmoid(capsule_acts_logits)

        if point_lists is None:
            inst_maps, point_lists, segmentation_lists = self.hough_routing(fg_pred, regressions, gt_seg)
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

                inst_capsule_poses = capsule_poses[i, :, y_coords, x_coords]  # (256, p)
                inst_capsule_poses = torch.transpose((inst_capsule_poses), 0, 1)  # (p, 256)

                if config.positional_encoding == True:
                    inst_points_mean = torch.mean(inst_points.float(), 0, keepdim=True)
                    inst_points_rel = inst_points - inst_points_mean  # gets the relative coordinates
                    y_coords_rel, x_coords_rel = inst_points_rel[0, :], inst_points_rel[1, :]
                    y_coords_rel, x_coords_rel = y_coords_rel / float(h / 16), x_coords_rel / float(
                        w / 16)  # Performs normalization between 0 and 1

                    if config.positional_encoding_type == 'addition':
                        inst_capsule_poses[:, -1] += x_coords_rel.cuda()
                        inst_capsule_poses[:, -2] += y_coords_rel.cuda()
                    elif config.positional_encoding_type == 'concat':
                        inst_capsule_poses = torch.cat((inst_capsule_poses, y_coords_rel.unsqueeze(1).float().cuda(), x_coords_rel.unsqueeze(1).float().cuda()), 1)

                inst_capsule_acts = capsule_acts[i, 0, y_coords, x_coords]  # (p, )

                out_capsule_poses, out_capsule_acts = self.transformer_routing(inst_capsule_poses,
                                                                               inst_capsule_acts)  # (34, F_out), (34, )

                class_outs.append(out_capsule_acts)

            class_outputs.append(torch.stack(class_outs) if len(class_outs) != 0 else [])

        # Should output center with shape (B, 1, H/16, W/16)
        # and regressions with shape(B, 2, H/16, W/16)
        return fg_pred, center, regressions, class_outputs, inst_maps, segmentation_lists

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)


class VotingModule(nn.Module):
    def __init__(self, n_caps_in, in_caps_dim, vote_dim):
        super(VotingModule, self).__init__()

        self.n_caps_in = n_caps_in
        self.in_caps_dim = in_caps_dim
        self.vote_dim = vote_dim

        self.vote_transform = nn.ModuleList([nn.Conv2d(in_caps_dim, vote_dim, 1) for _ in range(n_caps_in)])

    def forward(self, poses):
        """

        :param poses: Poses of shape (B, N_i*F_1, H, W)
        :return: Votes of shape (B, N_i*F_2, H, W)
        """

        b, f1, h, w = poses.shape
        assert f1 == self.n_caps_in*self.in_caps_dim

        poses = poses.view(b, self.n_caps_in, self.in_caps_dim, h, w)

        votes = []
        for i in range(self.n_caps_in):
            votes.append(self.vote_transform[i](poses[:, i]))
        votes = torch.stack(votes, 1)

        votes = votes.view(b, self.n_caps_in*self.vote_dim, h, w)

        return votes


class CapsuleModel2(nn.Module):
    def __init__(self, model_id, project_dir):
        super(CapsuleModel2, self).__init__()
        self.num_classes = config.n_classes
        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        self.resnet = ResNet50_OS16()  # NOTE! specify the type of ResNet here

        self.aspp = ASPP_Bottleneck()

        in_feats = 1280

        self.fg_decoder = fg_decoder(in_feats=in_feats, num_classes=self.num_classes)
        self.instance_decoder = inst_decoder(in_feats=in_feats)

        self.hough_routing = HoughRouting1()

        self.n_init_capsules = 8
        self.init_capsule_dim = 256//8
        self.conv1x1_poses = nn.Conv2d(in_feats, self.n_init_capsules*self.init_capsule_dim, 1)
        self.conv1x1_acts = nn.Conv2d(in_feats, self.n_init_capsules, 1)

        self.vote_dim = config.vote_dim
        self.vote_transform = VotingModule(self.n_init_capsules, self.init_capsule_dim, self.vote_dim)

        if config.positional_encoding == True:
            if config.positional_encoding_type == 'addition':
                self.pos_vote_transform = nn.Linear(self.vote_dim, self.vote_dim)  # Will need to add two to first argument if concatenating positional encoding
            elif config.positional_encoding_type == 'concat':
                self.pos_vote_transform = nn.Linear(self.vote_dim + 2, self.vote_dim)  # Will need to add two to first argument if concatenating positional encoding
        self.noise_scale = 4.0

        # self.transformer_routing = TransformerRouting(n_feats_in=self.vote_dim, n_caps_out=self.num_classes, output_dim=16, use_vote_transform=False)
        self.transformer_routing1 = TransformerRouting(n_feats_in=self.vote_dim, n_caps_out=64, output_dim=16, use_vote_transform=True)
        self.transformer_routing2 = TransformerRouting(n_feats_in=self.vote_dim, n_caps_out=self.num_classes, output_dim=16, use_vote_transform=False)

    def forward(self, x, point_lists=None, gt_seg=None, gt_reg=None):
        # (x has shape (batch_size, 3, h, w))
        h = x.size()[2]
        w = x.size()[3]

        # Encoder:
        feature_map, skip_8, skip_4 = self.resnet(x)  # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNe$

        feature_output = self.aspp(feature_map)  # (shape: (batch_size, 1280, h/16, w/16))

        # Decoder for semantic segmentation:
        fg_pred = self.fg_decoder(feature_output, skip_8, skip_4)
        fg_pred = F.sigmoid(fg_pred)

        # Decoder for instance segmentation:
        center, regressions = self.instance_decoder(feature_output, skip_8, skip_4)
        center = F.sigmoid(center)
        regressions = F.tanh(regressions)

        # Resizes the output maps
        fg_pred = F.upsample(fg_pred, size=(h, w), mode="bilinear")  # (shape: (batch_size, num_classes, h, w))
        center = F.upsample(center, size=(h, w), mode="bilinear")
        regressions = F.upsample(regressions, size=(h, w), mode="bilinear")
        regressions[:, 0] = regressions[:, 0] * w
        regressions[:, 1] = regressions[:, 1] * h

        # creates the first capsule layer
        capsule_poses = self.conv1x1_poses(feature_output)  # (batch_size, n_caps*caps_dim, h/16, w/16)
        capsule_acts_logits = self.conv1x1_acts(feature_output)   # (batch_size, n_caps, h/16, w/16)

        capsule_votes = self.vote_transform(capsule_poses)  # (batch_size, n_caps*vote_dim, h/16, w/16)

        if self.training:
            capsule_acts_logits += ((torch.rand(*capsule_acts_logits.shape) - 0.5) * self.noise_scale).cuda()  # adds noise to force network to learn binary activations
        capsule_acts = F.sigmoid(capsule_acts_logits)

        if point_lists is None:
            # inst_maps, point_lists, segmentation_lists = self.hough_routing(fg_pred, regressions, gt_seg)
            if gt_reg is None:
                inst_maps, point_lists, segmentation_lists = self.hough_routing(fg_pred, regressions, gt_seg)
            else:
                inst_maps, point_lists, segmentation_lists = self.hough_routing(fg_pred, gt_reg, gt_seg)
        # elif gt_reg is not None:
            # inst_maps, point_lists, segmentation_lists = self.hough_routing(fg_pred, gt_reg, gt_seg)
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

                inst_capsule_votes = capsule_votes[i, :, y_coords, x_coords]  # (n_caps*vote_dim, p)
                inst_capsule_votes = inst_capsule_votes.view(self.n_init_capsules, self.vote_dim, len(y_coords))  # (n_caps, vote_dim, p)
                inst_capsule_votes = torch.transpose(inst_capsule_votes, 1, 2).reshape(self.n_init_capsules*len(y_coords), self.vote_dim)  # (n_caps*p, vote_dim)

                #assert config.positional_encoding == False  # positional encoding will require more changes
                if config.positional_encoding == True:
                    inst_points_mean = torch.mean(inst_points.float(), 0, keepdim=True)
                    inst_points_rel = inst_points - inst_points_mean  # gets the relative coordinates
                    y_coords_rel, x_coords_rel = inst_points_rel[0, :], inst_points_rel[1, :]
                    y_coords_rel, x_coords_rel = y_coords_rel / float(h / 16), x_coords_rel / float(w / 16)  # Performs normalization between 0 and 1

                    # x_coords_rel and y_coords_rel should be of shape (p, )
                    x_coords_rel = x_coords_rel.unsqueeze(0).repeat(self.n_init_capsules, 1).reshape(self.n_init_capsules*len(y_coords), )  # makes x_coords_rel of shape (N*P, )
                    y_coords_rel = y_coords_rel.unsqueeze(0).repeat(self.n_init_capsules, 1).reshape(self.n_init_capsules*len(y_coords), )  # makes y_coords_rel of shape (N*P, )

                    if config.positional_encoding_type == 'addition':
                        inst_capsule_votes[:, -1] += x_coords_rel.cuda()
                        inst_capsule_votes[:, -2] += y_coords_rel.cuda()
                    elif config.positional_encoding_type == 'concat':
                        inst_capsule_votes = torch.cat((inst_capsule_votes, y_coords_rel.unsqueeze(1).float().cuda(), x_coords_rel.unsqueeze(1).float().cuda()), 1)

                    inst_capsule_votes = self.pos_vote_transform(inst_capsule_votes)


                inst_capsule_acts = capsule_acts[i, :, y_coords, x_coords]    # (n_caps, p)
                inst_capsule_acts = inst_capsule_acts.view(self.n_init_capsules*len(y_coords), )  # (n_caps*p, )

                # out_capsule_poses, out_capsule_acts = self.transformer_routing(inst_capsule_votes, inst_capsule_acts)  # (34, F_out), (34, )

                out_capsule_poses, out_capsule_acts = self.transformer_routing1(inst_capsule_votes, inst_capsule_acts)
                out_capsule_poses = out_capsule_poses.transpose(1, 0)
                out_capsule_poses, out_capsule_acts = self.transformer_routing2(out_capsule_poses, out_capsule_acts)

                class_outs.append(out_capsule_acts)

            class_outputs.append(torch.stack(class_outs) if len(class_outs) != 0 else [])

        # Should output center with shape (B, 1, H/16, W/16)
        # and regressions with shape(B, 2, H/16, W/16)
        return fg_pred, center, regressions, class_outputs, inst_maps, segmentation_lists

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)

