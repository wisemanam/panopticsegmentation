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
from capsules import PrimaryCaps

class VotingModule(nn.Module):
    def __init__(self, n_caps_in, in_caps_dim, vote_dim, kernel_dim=1, dilation=1, relu=False):
        super(VotingModule, self).__init__()

        self.n_caps_in = n_caps_in
        self.in_caps_dim = in_caps_dim
        self.vote_dim = vote_dim
        padding = (kernel_dim - 1 + (kernel_dim - 1)*(dilation - 1))//2
        self.vote_transform = nn.ModuleList([nn.Conv2d(in_caps_dim, vote_dim, kernel_dim, padding=padding, dilation=dilation) for _ in range(n_caps_in)])
        self.relu = relu

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

        if self.relu:
            return F.relu(votes)
        else:
            return votes



def resize_capsules(poses, acts, size):
    """

    :param poses: capsule poses or votes of shape (B, K1, H, W)
    :param acts: capsule activations or votes of shape (B, K2, H, W)
    :param size: tuple with height and width (h, w)
    :return: returns the poses and acts resized to have dimension (B, K1, h, w), (B, K2, h, w)
    """

    poses_up = F.upsample(poses, size=size, mode="bilinear")
    acts_up = F.upsample(acts, size=size, mode="bilinear")
    return poses_up, acts_up


def get_outputs_from_caps(fgbg_poses, fgbg_acts, regression_layer, input_size):
    h, w = input_size

    fg_pred = fgbg_acts[..., 0].unsqueeze(1)  # Shape (batch_size, 1, h/16, w/16)

    fg_poses = fgbg_poses[..., 0, :]
    regressions = regression_layer(fg_poses).permute(0, 3, 1, 2)
    regressions = F.tanh(regressions)

    # Resizes the output maps
    fg_pred = F.upsample(fg_pred, size=(h, w), mode="bilinear")  # (shape: (batch_size, num_classes, h, w))
    regressions = F.upsample(regressions, size=(h, w), mode="bilinear")
    regressions[:, 0] = regressions[:, 0] * w
    regressions[:, 1] = regressions[:, 1] * h

    return fg_pred, regressions
    
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

        self.hough_routing = HoughRouting1()

        self.n_init_capsules = [4, 4, config.n_init_capsules]
        self.init_capsule_dim = [4, 8, config.init_capsule_dim]

        self.primary_capsules_skip_4 = PrimaryCaps(256, self.n_init_capsules[0], 1, 1, self.init_capsule_dim[0])
        self.primary_capsules_skip_8 = PrimaryCaps(512, self.n_init_capsules[1], 1, 1, self.init_capsule_dim[1])
        self.primary_capsules = PrimaryCaps(in_feats, self.n_init_capsules[2], 1, 1, self.init_capsule_dim[2])

        self.vote_dim_seg = config.vote_dim_seg
        self.vote_transform_seg_skip_4 = VotingModule(self.n_init_capsules[0], self.init_capsule_dim[0], self.vote_dim_seg)
        self.vote_transform_seg_skip_8 = VotingModule(self.n_init_capsules[1], self.init_capsule_dim[1], self.vote_dim_seg)
        self.vote_transform_seg = VotingModule(self.n_init_capsules[2], self.init_capsule_dim[2], self.vote_dim_seg)

        self.vote_dim = config.vote_dim
        self.vote_transform_class = VotingModule(self.n_init_capsules[2], self.init_capsule_dim[2], self.vote_dim)

        if config.positional_encoding == True:
            if config.positional_encoding_type == 'addition':
                self.pos_vote_transform = nn.Linear(self.vote_dim, self.vote_dim)  # Will need to add two to first argument if concatenating positional encoding
            elif config.positional_encoding_type == 'concat':
                self.pos_vote_transform = nn.Linear(self.vote_dim + 2, self.vote_dim)  # Will need to add two to first argument if concatenating positional encoding

        self.noise_scale = 4.0

        self.transformer_routing = TransformerRouting(n_feats_in=self.vote_dim, n_caps_out=self.num_classes, output_dim=16, use_vote_transform=False, top_down_routing=config.use_top_down_routing)

        self.transformer_routing_seg = TransformerRouting(n_feats_in=self.vote_dim_seg, n_caps_out=2, hidden_dim=self.vote_dim_seg, output_dim=16, use_vote_transform=False, top_down_routing=config.use_top_down_routing)
        self.regression_linear = nn.Linear(16, 2)

    def forward(self, x, point_lists=None, gt_seg=None, gt_reg=None):
        # (x has shape (batch_size, 3, h, w))
        h = x.size()[2]
        w = x.size()[3]

        # Encoder:
        feature_map, skip_8, skip_4 = self.resnet(x)  # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNe$

        feature_output = self.aspp(feature_map)  # (shape: (batch_size, 1280, h/16, w/16))

        # creates the first capsule layer
        capsule_poses_s4, capsule_acts_s4 = self.primary_capsules_skip_4(skip_4)
        capsule_poses_s8, capsule_acts_s8 = self.primary_capsules_skip_8(skip_8)

        capsule_poses, capsule_acts = self.primary_capsules(feature_output)

        capsule_votes_seg_s4 = self.vote_transform_seg_skip_4(capsule_poses_s4)  # (batch_size, n_caps*vote_dim, h/4, w/4)
        capsule_votes_seg_s8 = self.vote_transform_seg_skip_8(capsule_poses_s8)  # (batch_size, n_caps*vote_dim, h/8, w/8)
        capsule_votes_seg = self.vote_transform_seg(capsule_poses)  # (batch_size, n_caps*vote_dim, h/16, w/16)

        capsule_votes_seg_up_s8 = F.upsample(capsule_votes_seg_s8, size=(h//4, w//4), mode="bilinear")
        capsule_acts_up_s8 = F.upsample(capsule_acts_s8, size=(h//4, w//4), mode="bilinear")

        capsule_votes_seg_up = F.upsample(capsule_votes_seg, size=(h//4, w//4), mode="bilinear")
        capsule_acts_up = F.upsample(capsule_acts, size=(h//4, w//4), mode="bilinear")

        b_size, _, h_new, w_new = capsule_votes_seg_up.shape

        capsule_votes_seg_s4 = capsule_votes_seg_s4.permute(0, 2, 3, 1).view(b_size, h_new, w_new, self.n_init_capsules[0], self.vote_dim_seg) # (B, h', w', 4, 32)
        capsule_votes_seg_s8 = capsule_votes_seg_up_s8.permute(0, 2, 3, 1).view(b_size, h_new, w_new, self.n_init_capsules[1], self.vote_dim_seg) # (B, h', w', 8, 32)
        capsule_votes_seg = capsule_votes_seg_up.permute(0, 2, 3, 1).view(b_size, h_new, w_new, self.n_init_capsules[2], self.vote_dim_seg) # (B, h', w', 32, 32)
        capsule_votes_seg = torch.cat((capsule_votes_seg, capsule_votes_seg_s8, capsule_votes_seg_s4), -2) # (B, h, w, 16, 32)
        capsule_acts_seg = torch.cat((capsule_acts_up, capsule_acts_up_s8, capsule_acts_s4), 1).permute(0, 2, 3, 1) # (B, h, w, 16)

        fgbg_poses, fgbg_acts = self.transformer_routing_seg(capsule_votes_seg, capsule_acts_seg)

        fg_pred = fgbg_acts[..., 0].unsqueeze(1)  # Shape (batch_size, 1, h/16, w/16)

        fg_poses = fgbg_poses[..., 0, :]
        regressions = self.regression_linear(fg_poses).permute(0, 3, 1, 2)
        regressions = F.tanh(regressions)

        # Resizes the output maps
        fg_pred = F.upsample(fg_pred, size=(h, w), mode="bilinear")  # (shape: (batch_size, num_classes, h, w))
        center = torch.zeros_like(fg_pred)  # F.upsample(center, size=(h, w), mode="bilinear")
        regressions = F.upsample(regressions, size=(h, w), mode="bilinear")
        regressions[:, 0] = regressions[:, 0] * w
        regressions[:, 1] = regressions[:, 1] * h

        capsule_votes_class = self.vote_transform_class(capsule_poses)  # (batch_size, n_caps*vote_dim, h/16, w/16)

        if point_lists is None:
            # inst_maps, point_lists, segmentation_lists = self.hough_routing(fg_pred, regressions, gt_seg)
            if gt_reg is None:
                inst_maps, point_lists, segmentation_lists = self.hough_routing(fg_pred, regressions, gt_seg)
            else:
                inst_maps, point_lists, segmentation_lists = self.hough_routing(fg_pred, gt_reg, gt_seg)
        else:
            inst_maps = []
            segmentation_lists = []

        # class_poses, class_acts = torch.zeros(batch_size, h_new, w_new, 8, 16), torch.zeros(batch_size, h_new, w_new, 8, 1)

        class_outputs = []
        for i, point_list in enumerate(point_lists):

            class_outs = []
            for inst_points in point_list:
                # gather capsules corresponding to inst_points
                inst_points = torch.unique(inst_points // 16, dim=1)

                y_coords, x_coords = inst_points[0, :], inst_points[1, :]

                inst_capsule_votes = capsule_votes_class[i, :, y_coords, x_coords]  # (n_caps*vote_dim, p)
                inst_capsule_votes = inst_capsule_votes.view(self.n_init_capsules[2], self.vote_dim, len(y_coords))  # (n_caps, vote_dim, p)
                inst_capsule_votes = torch.transpose(inst_capsule_votes, 1, 2).reshape(self.n_init_capsules[2]*len(y_coords), self.vote_dim)  # (n_caps*p, vote_dim)

                #assert config.positional_encoding == False  # positional encoding will require more changes
                if config.positional_encoding == True:
                    inst_points_mean = torch.mean(inst_points.float(), 0, keepdim=True)
                    inst_points_rel = inst_points - inst_points_mean  # gets the relative coordinates
                    y_coords_rel, x_coords_rel = inst_points_rel[0, :], inst_points_rel[1, :]
                    y_coords_rel, x_coords_rel = y_coords_rel / float(h / 16), x_coords_rel / float(w / 16)  # Performs normalization between 0 and 1

                    # x_coords_rel and y_coords_rel should be of shape (p, )
                    x_coords_rel = x_coords_rel.unsqueeze(0).repeat(self.n_init_capsules[2], 1).reshape(self.n_init_capsules[2]*len(y_coords), )  # makes x_coords_rel of shape (N*P, )
                    y_coords_rel = y_coords_rel.unsqueeze(0).repeat(self.n_init_capsules[2], 1).reshape(self.n_init_capsules[2]*len(y_coords), )  # makes y_coords_rel of shape (N*P, )

                    if config.positional_encoding_type == 'addition':
                        inst_capsule_votes[:, -1] += x_coords_rel.cuda()
                        inst_capsule_votes[:, -2] += y_coords_rel.cuda()
                    elif config.positional_encoding_type == 'concat':
                        inst_capsule_votes = torch.cat((inst_capsule_votes, y_coords_rel.unsqueeze(1).float().cuda(), x_coords_rel.unsqueeze(1).float().cuda()), 1)

                    inst_capsule_votes = self.pos_vote_transform(inst_capsule_votes)

                inst_capsule_acts = capsule_acts[i, :, y_coords, x_coords]    # (n_caps, p)
                inst_capsule_acts = inst_capsule_acts.view(self.n_init_capsules[2]*len(y_coords), )  # (n_caps*p, )

                out_capsule_poses, out_capsule_acts = self.transformer_routing(inst_capsule_votes, inst_capsule_acts)  # (34, F_out), (34, )
                # out_capsule_poses, out_capsule_acts = self.transformer_routing1(inst_capsule_votes, inst_capsule_acts)
                # out_capsule_poses, out_capsule_acts = self.transformer_routing2(out_capsule_poses, out_capsule_acts)

                class_outs.append(out_capsule_acts)

            class_outputs.append(torch.stack(class_outs) if len(class_outs) != 0 else [])

        # Should output center with shape (B, 1, H/16, W/16)
        # and regressions with shape(B, 2, H/16, W/16)
        return fg_pred, regressions, class_outputs, inst_maps, segmentation_lists

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)


class CapsuleModel5(nn.Module):
    def __init__(self, model_id, project_dir):
        super(CapsuleModel5, self).__init__()
        self.num_classes = config.n_classes
        self.num_inst_classes = 34
        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        self.resnet = ResNet50_OS16()  # NOTE! specify the type of ResNet here

        self.aspp = ASPP_Bottleneck()

        self.noise_scale = 4.0
        in_feats = 1280

        self.hough_routing = HoughRouting1()

        self.n_init_capsules = [4, 4, 8]
        self.init_capsule_dim = [4, 8, 32]

        self.primary_capsules_skip_4 = PrimaryCaps(256, self.n_init_capsules[0], 1, 1, self.init_capsule_dim[0])
        self.primary_capsules_skip_8 = PrimaryCaps(512, self.n_init_capsules[1], 1, 1, self.init_capsule_dim[1])
        self.primary_capsules = PrimaryCaps(in_feats, self.n_init_capsules[2], 1, 1, self.init_capsule_dim[2])

        self.vote_dim_seg = 32
        self.vote_transform_seg_skip_4 = VotingModule(self.n_init_capsules[0], self.init_capsule_dim[0], self.vote_dim_seg)
        self.vote_transform_seg_skip_8 = VotingModule(self.n_init_capsules[1], self.init_capsule_dim[1], self.vote_dim_seg)
        self.vote_transform_seg = VotingModule(self.n_init_capsules[2], self.init_capsule_dim[2], self.vote_dim_seg)

        self.vote_dim = config.vote_dim
        self.vote_transform_class = VotingModule(self.n_init_capsules[2], self.init_capsule_dim[2], self.vote_dim)

        if config.positional_encoding == True:
            if config.positional_encoding_type == 'addition':
                self.pos_vote_transform = nn.Linear(self.vote_dim, self.vote_dim)  # Will need to add two to first argument if concatenating positional encoding
            elif config.positional_encoding_type == 'concat':
                self.pos_vote_transform = nn.Linear(self.vote_dim + 2, self.vote_dim)  # Will need to add two to first argument if concatenating positional encoding

        self.transformer_routing = TransformerRouting(n_feats_in=self.vote_dim, n_caps_out=self.num_inst_classes, output_dim=16, use_vote_transform=False)

        self.transformer_routing_seg = TransformerRouting(n_feats_in=self.vote_dim_seg, n_caps_out=2, hidden_dim=self.vote_dim_seg, output_dim=16, use_vote_transform=False)
        self.regression_linear = nn.Linear(16, 2)

        self.vote_transform_seg2 = VotingModule(2 + self.num_inst_classes, 16, 16, kernel_dim=3, dilation=3)
        self.vote_transform_seg3 = VotingModule(2 + self.num_inst_classes, 16, self.vote_dim_seg, kernel_dim=3, dilation=3)

        self.transformer_routing_seg2 = TransformerRouting(n_feats_in=self.vote_dim_seg, n_caps_out=2, hidden_dim=self.vote_dim_seg, output_dim=16, use_vote_transform=False)
        self.regression_linear2 = nn.Linear(16, 2)

        self.vote_transform_class2 = VotingModule(self.n_init_capsules[2], self.init_capsule_dim[2], self.vote_dim)

        if config.positional_encoding == True:
            if config.positional_encoding_type == 'addition':
                self.pos_vote_transform2 = nn.Linear(self.vote_dim, self.vote_dim)  # Will need to add two to first argument if concatenating positional encoding
            elif config.positional_encoding_type == 'concat':
                self.pos_vote_transform2 = nn.Linear(self.vote_dim + 2, self.vote_dim)  # Will need to add two to first argument if concatenating positional encoding

        self.transformer_routing2 = TransformerRouting(n_feats_in=self.vote_dim, n_caps_out=self.num_inst_classes, output_dim=16, use_vote_transform=False)

    def get_primary_and_fg_capsules(self, feature_output, skip_8, skip_4, input_size):
        h, w = input_size

        capsule_poses, capsule_acts = self.primary_capsules(feature_output)

        capsule_poses_s4, capsule_acts_s4 = self.primary_capsules_skip_4(skip_4)
        capsule_poses_s8, capsule_acts_s8 = self.primary_capsules_skip_8(skip_8)

        capsule_votes_seg_s4 = self.vote_transform_seg_skip_4(capsule_poses_s4)  # (batch_size, n_caps*vote_dim, h/4, w/4)
        capsule_votes_seg_s8 = self.vote_transform_seg_skip_8(capsule_poses_s8)  # (batch_size, n_caps*vote_dim, h/8, w/8)
        capsule_votes_seg = self.vote_transform_seg(capsule_poses)  # (batch_size, n_caps*vote_dim, h/16, w/16)

        capsule_votes_seg_up_s8, capsule_acts_up_s8 = resize_capsules(capsule_votes_seg_s8, capsule_acts_s8, size=(h // 4, w // 4))
        capsule_votes_seg_up, capsule_acts_up = resize_capsules(capsule_votes_seg, capsule_acts, size=(h // 4, w // 4))

        b_size, _, h_new, w_new = capsule_votes_seg_up.shape

        capsule_votes_seg_s4 = capsule_votes_seg_s4.permute(0, 2, 3, 1).view(b_size, h_new, w_new, self.n_init_capsules[0], self.vote_dim_seg)  # (B, h', w', 4, 32)
        capsule_votes_seg_s8 = capsule_votes_seg_up_s8.permute(0, 2, 3, 1).view(b_size, h_new, w_new, self.n_init_capsules[1], self.vote_dim_seg)  # (B, h', w', 8, 32)
        capsule_votes_seg = capsule_votes_seg_up.permute(0, 2, 3, 1).view(b_size, h_new, w_new, self.n_init_capsules[2], self.vote_dim_seg)  # (B, h', w', 32, 32)

        capsule_votes_seg = torch.cat((capsule_votes_seg, capsule_votes_seg_s8, capsule_votes_seg_s4), -2)  # (B, h, w, 16, 32)
        capsule_acts_seg = torch.cat((capsule_acts_up, capsule_acts_up_s8, capsule_acts_s4), 1).permute(0, 2, 3, 1)  # (B, h, w, 16)

        fgbg_poses, fgbg_acts = self.transformer_routing_seg(capsule_votes_seg, capsule_acts_seg)  # (B, H_new, W_new, 2, 16), (B, H_new, W_new, 2)

        return (capsule_poses, capsule_acts), (fgbg_poses, fgbg_acts)

    def create_inst_maps(self, point_lists, gt_reg, gt_seg, fg_pred, regressions):
        if point_lists is None:
            if gt_reg is None:
                inst_maps, point_lists, segmentation_lists = self.hough_routing(fg_pred, regressions, gt_seg)
            else:
                inst_maps, point_lists, segmentation_lists = self.hough_routing(fg_pred, gt_reg, gt_seg)
        else:
            inst_maps = []
            segmentation_lists = []

        return point_lists, inst_maps, segmentation_lists

    def scatter_capsules(self, point_lists, capsule_votes_inst, capsule_acts, input_size, instance_scale):
        h, w = input_size
        b_size, _, h_inp, w_inp = capsule_votes_inst.shape
        assert h//h_inp == w/w_inp
        capsule_scale = h//h_inp

        instance_poses = torch.zeros((b_size, h//instance_scale, w//instance_scale, self.num_inst_classes, 16))
        instance_acts = torch.zeros((b_size, h//instance_scale, w//instance_scale, self.num_inst_classes))

        class_outputs = []
        for i, point_list in enumerate(point_lists):

            class_outs = []
            for inst_points in point_list:
                # gather capsules corresponding to inst_points
                inst_points_down16 = torch.unique(inst_points // capsule_scale, dim=1)

                y_coords, x_coords = inst_points_down16[0, :], inst_points_down16[1, :]

                inst_capsule_votes = capsule_votes_inst[i, :, y_coords, x_coords]  # (n_caps*vote_dim, p)
                inst_capsule_votes = inst_capsule_votes.view(self.n_init_capsules[2], self.vote_dim, len(y_coords))  # (n_caps, vote_dim, p)
                inst_capsule_votes = torch.transpose(inst_capsule_votes, 1, 2).reshape(self.n_init_capsules[2] * len(y_coords), self.vote_dim)  # (n_caps*p, vote_dim)

                if config.positional_encoding == True:
                    inst_points_mean = torch.mean(inst_points.float(), 0, keepdim=True)
                    inst_points_rel = inst_points - inst_points_mean  # gets the relative coordinates
                    y_coords_rel, x_coords_rel = inst_points_rel[0, :], inst_points_rel[1, :]
                    y_coords_rel, x_coords_rel = y_coords_rel / float(h / capsule_scale), x_coords_rel / float(w / capsule_scale)  # Performs normalization between 0 and 1

                    # x_coords_rel and y_coords_rel should be of shape (p, )
                    x_coords_rel = x_coords_rel.unsqueeze(0).repeat(self.n_init_capsules[2], 1).reshape(self.n_init_capsules[2] * len(y_coords), )  # makes x_coords_rel of shape (N*P, )
                    y_coords_rel = y_coords_rel.unsqueeze(0).repeat(self.n_init_capsules[2], 1).reshape(self.n_init_capsules[2] * len(y_coords), )  # makes y_coords_rel of shape (N*P, )

                    if config.positional_encoding_type == 'addition':
                        inst_capsule_votes[:, -1] += x_coords_rel.cuda()
                        inst_capsule_votes[:, -2] += y_coords_rel.cuda()
                    elif config.positional_encoding_type == 'concat':
                        inst_capsule_votes = torch.cat((inst_capsule_votes, y_coords_rel.unsqueeze(1).float().cuda(), x_coords_rel.unsqueeze(1).float().cuda()), 1)

                    inst_capsule_votes = self.pos_vote_transform(inst_capsule_votes)

                inst_capsule_acts = capsule_acts[i, :, y_coords, x_coords]  # (n_caps, p)
                inst_capsule_acts = inst_capsule_acts.view(self.n_init_capsules[2] * len(y_coords), )  # (n_caps*p, )

                out_capsule_poses, out_capsule_acts = self.transformer_routing(inst_capsule_votes, inst_capsule_acts)  # (34, F_out), (34, )

                inst_points_down4 = torch.unique(inst_points // instance_scale, dim=1)
                y_coords, x_coords = inst_points_down4[0, :], inst_points_down4[1, :]

                instance_poses[i, y_coords, x_coords] = out_capsule_poses.cpu()
                instance_acts[i, y_coords, x_coords] = out_capsule_acts.cpu()

                class_outs.append(out_capsule_acts)

            class_outputs.append(torch.stack(class_outs) if len(class_outs) != 0 else [])

        return instance_poses, instance_acts, class_outputs

    def scatter_capsules2(self, point_lists, capsule_votes_inst, capsule_acts, input_size):
        h, w = input_size
        b_size, _, h_inp, w_inp = capsule_votes_inst.shape
        assert h//h_inp == w/w_inp
        capsule_scale = h//h_inp

        class_outputs = []
        for i, point_list in enumerate(point_lists):

            class_outs = []
            for inst_points in point_list:
                # gather capsules corresponding to inst_points
                inst_points_down16 = torch.unique(inst_points // capsule_scale, dim=1)

                y_coords, x_coords = inst_points_down16[0, :], inst_points_down16[1, :]

                inst_capsule_votes = capsule_votes_inst[i, :, y_coords, x_coords]  # (n_caps*vote_dim, p)
                inst_capsule_votes = inst_capsule_votes.view(self.n_init_capsules[2], self.vote_dim, len(y_coords))  # (n_caps, vote_dim, p)
                inst_capsule_votes = torch.transpose(inst_capsule_votes, 1, 2).reshape(self.n_init_capsules[2] * len(y_coords), self.vote_dim)  # (n_caps*p, vote_dim)

                if config.positional_encoding == True:
                    inst_points_mean = torch.mean(inst_points.float(), 0, keepdim=True)
                    inst_points_rel = inst_points - inst_points_mean  # gets the relative coordinates
                    y_coords_rel, x_coords_rel = inst_points_rel[0, :], inst_points_rel[1, :]
                    y_coords_rel, x_coords_rel = y_coords_rel / float(h / capsule_scale), x_coords_rel / float(w / capsule_scale)  # Performs normalization between 0 and 1

                    # x_coords_rel and y_coords_rel should be of shape (p, )
                    x_coords_rel = x_coords_rel.unsqueeze(0).repeat(self.n_init_capsules[2], 1).reshape(self.n_init_capsules[2] * len(y_coords), )  # makes x_coords_rel of shape (N*P, )
                    y_coords_rel = y_coords_rel.unsqueeze(0).repeat(self.n_init_capsules[2], 1).reshape(self.n_init_capsules[2] * len(y_coords), )  # makes y_coords_rel of shape (N*P, )

                    if config.positional_encoding_type == 'addition':
                        inst_capsule_votes[:, -1] += x_coords_rel.cuda()
                        inst_capsule_votes[:, -2] += y_coords_rel.cuda()
                    elif config.positional_encoding_type == 'concat':
                        inst_capsule_votes = torch.cat((inst_capsule_votes, y_coords_rel.unsqueeze(1).float().cuda(), x_coords_rel.unsqueeze(1).float().cuda()), 1)

                    inst_capsule_votes = self.pos_vote_transform2(inst_capsule_votes)

                inst_capsule_acts = capsule_acts[i, :, y_coords, x_coords]  # (n_caps, p)
                inst_capsule_acts = inst_capsule_acts.view(self.n_init_capsules[2] * len(y_coords), )  # (n_caps*p, )

                out_capsule_poses, out_capsule_acts = self.transformer_routing2(inst_capsule_votes, inst_capsule_acts)  # (34, F_out), (34, )

                class_outs.append(out_capsule_acts)

            class_outputs.append(torch.stack(class_outs) if len(class_outs) != 0 else [])

        return class_outputs

    def forward(self, x, point_lists=None, gt_seg=None, gt_reg=None, two_stage=True):
        # (x has shape (batch_size, 3, h, w))
        _, _, h, w = x.shape

        fg_preds, reg_preds, class_preds, inst_maps_preds, seg_list_preds = [], [], [], [], []

        # Encoder:
        feature_map, skip_8, skip_4 = self.resnet(x)  # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNe$

        feature_output = self.aspp(feature_map)  # (shape: (batch_size, 1280, h/16, w/16))

        # creates primary capsule layer and the initial foreground-background capsules
        x = self.get_primary_and_fg_capsules(feature_output, skip_8, skip_4, (h, w))
        (primary_poses, primary_acts), (fgbg_poses, fgbg_acts) = x

        # gets the network outputs (foreground segmentations and regressions) from foreground-background capsules
        fg_pred, regressions = get_outputs_from_caps(fgbg_poses, fgbg_acts, self.regression_linear, (h, w))

        # Uses hough-routing method to get instance maps from the foreground segmentations and regressions
        point_lists1, inst_maps, segmentation_lists = self.create_inst_maps(point_lists, gt_reg, gt_seg, fg_pred, regressions)

        capsule_votes_inst = self.vote_transform_class(primary_poses)  # (batch_size, n_caps*vote_dim, h/16, w/16)

        b_size, h_new, w_new, _, _ = fgbg_poses.shape

        # performs scatter capsule operation
        x = self.scatter_capsules(point_lists1, capsule_votes_inst, primary_acts, (h, w), instance_scale=4)
        instance_poses, instance_acts, class_outputs = x

        # appends network outputs
        fg_preds.append(fg_pred)
        reg_preds.append(regressions)
        class_preds.append([class8to34(class_output) for class_output in class_outputs])
        inst_maps_preds.append(inst_maps)
        seg_list_preds.append(segmentation_lists)
        
        if two_stage:
            new_capsules_poses = torch.cat((fgbg_poses.cuda(), instance_poses.cuda()), -2)
            new_capsules_acts = torch.cat((fgbg_acts.cuda(), instance_acts.cuda()), -1)

            new_capsules_poses = new_capsules_poses.view(b_size, h_new, w_new, -1).permute(0, 3, 1, 2)
            capsule_votes_seg2 = self.vote_transform_seg2(new_capsules_poses)  # (batch_size, n_caps*vote_dim, h/4, w/4)
            capsule_votes_seg3 = self.vote_transform_seg3(capsule_votes_seg2)
            capsule_votes_seg3 = capsule_votes_seg3.permute(0, 2, 3, 1).view(b_size, h_new, w_new, self.num_inst_classes+2, self.vote_dim_seg)
 
            fgbg_poses, fgbg_acts = self.transformer_routing_seg(capsule_votes_seg3, new_capsules_acts)  # (B, H_new, W_new, 2, 16), (B, H_new, W_new, 2)

            fg_pred, regressions = get_outputs_from_caps(fgbg_poses, fgbg_acts, self.regression_linear2, (h, w))

            point_lists2, inst_maps, segmentation_lists = self.create_inst_maps(point_lists, gt_reg, gt_seg, fg_pred, regressions)

            capsule_votes_inst2 = self.vote_transform_class2(primary_poses)

            class_outputs = self.scatter_capsules2(point_lists2, capsule_votes_inst2, primary_acts, (h, w))

            # appends network outputs
            fg_preds.append(fg_pred)
            reg_preds.append(regressions)
            class_preds.append([class8to34(class_output) for class_output in class_outputs])
            inst_maps_preds.append(inst_maps)
            seg_list_preds.append(segmentation_lists)

        # Should output center with shape (B, 1, H/16, W/16)
        # and regressions with shape(B, 2, H/16, W/16)
        
        return fg_preds, reg_preds, class_preds, inst_maps_preds, seg_list_preds

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)

def class8to34(y):
    # y should be shape (N, 8)
    # output should be (N, 34)
    if len(y) == 0:
        return y
    if y.shape[1] == 34:
        return y
    out = torch.zeros_like(y[:, :1]).repeat(1, 34)
    out[:, -3:] = y[:, -3:]
    out[:, -10:-5] = y[:, :5]
    
    return out


if __name__ == '__main__':
    tester = CapsuleModel5('CapsuleModel5', 'SimpleSegmentation/').cuda()

    inp = torch.zeros((5, 3, 512, 1024)).cuda()

    x = tester(inp)

class CapsuleModel6(nn.Module):
    def __init__(self, model_id, project_dir):
        super(CapsuleModel6, self).__init__()
        self.num_classes = config.n_classes
        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        self.resnet = ResNet50_OS16()  # NOTE! specify the type of ResNet here

        self.aspp = ASPP_Bottleneck()

        in_feats = 1280

        self.hough_routing = HoughRouting1()

        self.n_init_capsules = [4, 4, 8]
        self.init_capsule_dim = [4, 8, 32]
        
        self.num_inst_classes = 8

        self.primary_capsules_skip_4 = PrimaryCaps(256, self.n_init_capsules[0], 1, 1, self.init_capsule_dim[0])
        self.primary_capsules_skip_8 = PrimaryCaps(512, self.n_init_capsules[1], 1, 1, self.init_capsule_dim[1])
        self.primary_capsules = PrimaryCaps(in_feats, self.n_init_capsules[2], 1, 1, self.init_capsule_dim[2])

        self.vote_dim_seg = 32
        self.vote_transform_seg_skip_4 = VotingModule(self.n_init_capsules[0], self.init_capsule_dim[0], self.vote_dim_seg)
        self.vote_transform_seg_skip_8 = VotingModule(self.n_init_capsules[1], self.init_capsule_dim[1], self.vote_dim_seg)
        self.vote_transform_seg = VotingModule(self.n_init_capsules[2], self.init_capsule_dim[2], self.vote_dim_seg)

        self.vote_dim = config.vote_dim
        self.vote_transform_class = VotingModule(self.n_init_capsules[2], self.init_capsule_dim[2], self.vote_dim)

        if config.positional_encoding == True:
            if config.positional_encoding_type == 'addition':
                self.pos_vote_transform = nn.Linear(self.vote_dim, self.vote_dim)  # Will need to add two to first argument if concatenating positional encoding
            elif config.positional_encoding_type == 'concat':
                self.pos_vote_transform = nn.Linear(self.vote_dim + 2, self.vote_dim)  # Will need to add two to first argument if concatenating positional encoding
        self.noise_scale = 4.0

        self.transformer_routing = TransformerRouting(n_feats_in=self.vote_dim, n_caps_out=self.num_inst_classes, output_dim=16, use_vote_transform=False)

        self.transformer_routing_seg = TransformerRouting(n_feats_in=self.vote_dim_seg, n_caps_out=2, hidden_dim=self.vote_dim_seg, output_dim=16, use_vote_transform=False)
        self.regression_linear = nn.Linear(16, 2)
        
        self.vote_transform_seg2 = VotingModule(2 + self.num_inst_classes, 16, 16, kernel_dim=3, dilation=3)
        self.vote_transform_seg3 = VotingModule(2 + self.num_inst_classes, 16, self.vote_dim_seg, kernel_dim=3, dilation=3)

        self.transformer_routing_seg2 = TransformerRouting(n_feats_in=self.vote_dim_seg, n_caps_out=2, hidden_dim=self.vote_dim_seg, output_dim=16, use_vote_transform=False)
        self.regression_linear2 = nn.Linear(16, 2)

        self.vote_transform_class2 = VotingModule(self.n_init_capsules[2], self.init_capsule_dim[2], self.vote_dim)

        if config.positional_encoding == True:
            if config.positional_encoding_type == 'addition':
                self.pos_vote_transform2 = nn.Linear(self.vote_dim, self.vote_dim)  # Will need to add two to first argument if concatenating positional encoding
            elif config.positional_encoding_type == 'concat':
                self.pos_vote_transform2 = nn.Linear(self.vote_dim + 2, self.vote_dim)  # Will need to add two to first argument if concatenating positional encoding

        self.transformer_routing2 = TransformerRouting(n_feats_in=self.vote_dim, n_caps_out=self.num_inst_classes, output_dim=16, use_vote_transform=False)
        
    def forward(self, x, point_lists=None, gt_seg=None, gt_reg=None, two_stage=False):
        # (x has shape (batch_size, 3, h, w))
        h = x.size()[2]
        w = x.size()[3]

        # Encoder:
        feature_map, skip_8, skip_4 = self.resnet(x)  # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNe$

        feature_output = self.aspp(feature_map)  # (shape: (batch_size, 1280, h/16, w/16))

        # creates the first capsule layer
        capsule_poses_s4, capsule_acts_s4 = self.primary_capsules_skip_4(skip_4)
        capsule_poses_s8, capsule_acts_s8 = self.primary_capsules_skip_8(skip_8)

        capsule_poses, capsule_acts = self.primary_capsules(feature_output)

        capsule_votes_seg_s4 = self.vote_transform_seg_skip_4(capsule_poses_s4)  # (batch_size, n_caps*vote_dim, h/4, w/4)
        capsule_votes_seg_s8 = self.vote_transform_seg_skip_8(capsule_poses_s8)  # (batch_size, n_caps*vote_dim, h/8, w/8)
        capsule_votes_seg = self.vote_transform_seg(capsule_poses)  # (batch_size, n_caps*vote_dim, h/16, w/16)

        capsule_votes_seg_up_s8 = F.upsample(capsule_votes_seg_s8, size=(h//4, w//4), mode="bilinear")
        capsule_acts_up_s8 = F.upsample(capsule_acts_s8, size=(h//4, w//4), mode="bilinear")

        capsule_votes_seg_up = F.upsample(capsule_votes_seg, size=(h//4, w//4), mode="bilinear")
        capsule_acts_up = F.upsample(capsule_acts, size=(h//4, w//4), mode="bilinear")

        b_size, _, h_new, w_new = capsule_votes_seg_up.shape

        capsule_votes_seg_s4 = capsule_votes_seg_s4.permute(0, 2, 3, 1).view(b_size, h_new, w_new, self.n_init_capsules[0], self.vote_dim_seg) # (B, h', w', 4, 32)
        capsule_votes_seg_s8 = capsule_votes_seg_up_s8.permute(0, 2, 3, 1).view(b_size, h_new, w_new, self.n_init_capsules[1], self.vote_dim_seg) # (B, h', w', 8, 32)
        capsule_votes_seg = capsule_votes_seg_up.permute(0, 2, 3, 1).view(b_size, h_new, w_new, self.n_init_capsules[2], self.vote_dim_seg) # (B, h', w', 32, 32)
        capsule_votes_seg = torch.cat((capsule_votes_seg, capsule_votes_seg_s8, capsule_votes_seg_s4), -2) # (B, h, w, 16, 32)
        capsule_acts_seg = torch.cat((capsule_acts_up, capsule_acts_up_s8, capsule_acts_s4), 1).permute(0, 2, 3, 1) # (B, h, w, 16)

        fgbg_poses, fgbg_acts = self.transformer_routing_seg(capsule_votes_seg, capsule_acts_seg)

        fg_pred = fgbg_acts[..., 0].unsqueeze(1)  # Shape (batch_size, 1, h/16, w/16)

        fg_poses = fgbg_poses[..., 0, :]
        regressions = self.regression_linear(fg_poses).permute(0, 3, 1, 2)
        regressions = F.tanh(regressions)

        # Resizes the output maps
        fg_pred = F.upsample(fg_pred, size=(h, w), mode="bilinear")  # (shape: (batch_size, num_classes, h, w))
        center = torch.zeros_like(fg_pred)  # F.upsample(center, size=(h, w), mode="bilinear")
        regressions = F.upsample(regressions, size=(h, w), mode="bilinear")
        regressions[:, 0] = regressions[:, 0] * w
        regressions[:, 1] = regressions[:, 1] * h

        capsule_votes_class = self.vote_transform_class(capsule_poses)  # (batch_size, n_caps*vote_dim, h/16, w/16)

        if point_lists is None:
            # inst_maps, point_lists, segmentation_lists = self.hough_routing(fg_pred, regressions, gt_seg)
            if gt_reg is None:
                inst_maps, point_lists1, segmentation_lists = self.hough_routing(fg_pred, regressions, gt_seg)
            else:
                inst_maps, point_lists1, segmentation_lists = self.hough_routing(fg_pred, gt_reg, gt_seg)
        else:
            inst_maps = []
            segmentation_lists = []
            point_lists1 = point_lists
        class_outputs = []
        instance_poses = torch.zeros((b_size, h//4, w//4, self.num_inst_classes, 16))
        instance_acts = torch.zeros((b_size, h//4, w//4, self.num_inst_classes))
        
        for i, point_list in enumerate(point_lists1):

            class_outs = []
            for inst_points in point_list:
                # gather capsules corresponding to inst_points
                inst_points = torch.unique(inst_points // 16, dim=1)

                y_coords, x_coords = inst_points[0, :], inst_points[1, :]

                inst_capsule_votes = capsule_votes_class[i, :, y_coords, x_coords]  # (n_caps*vote_dim, p)
                inst_capsule_votes = inst_capsule_votes.view(self.n_init_capsules[2], self.vote_dim, len(y_coords))  # (n_caps, vote_dim, p)
                inst_capsule_votes = torch.transpose(inst_capsule_votes, 1, 2).reshape(self.n_init_capsules[2]*len(y_coords), self.vote_dim)  # (n_caps*p, vote_dim)

                #assert config.positional_encoding == False  # positional encoding will require more changes
                if config.positional_encoding == True:
                    inst_points_mean = torch.mean(inst_points.float(), 0, keepdim=True)
                    inst_points_rel = inst_points - inst_points_mean  # gets the relative coordinates
                    y_coords_rel, x_coords_rel = inst_points_rel[0, :], inst_points_rel[1, :]
                    y_coords_rel, x_coords_rel = y_coords_rel / float(h / 16), x_coords_rel / float(w / 16)  # Performs normalization between 0 and 1

                    # x_coords_rel and y_coords_rel should be of shape (p, )
                    x_coords_rel = x_coords_rel.unsqueeze(0).repeat(self.n_init_capsules[2], 1).reshape(self.n_init_capsules[2]*len(y_coords), )  # makes x_coords_rel of shape (N*P, )
                    y_coords_rel = y_coords_rel.unsqueeze(0).repeat(self.n_init_capsules[2], 1).reshape(self.n_init_capsules[2]*len(y_coords), )  # makes y_coords_rel of shape (N*P, )

                    if config.positional_encoding_type == 'addition':
                        inst_capsule_votes[:, -1] += x_coords_rel.cuda()
                        inst_capsule_votes[:, -2] += y_coords_rel.cuda()
                    elif config.positional_encoding_type == 'concat':
                        inst_capsule_votes = torch.cat((inst_capsule_votes, y_coords_rel.unsqueeze(1).float().cuda(), x_coords_rel.unsqueeze(1).float().cuda()), 1)

                    inst_capsule_votes = self.pos_vote_transform(inst_capsule_votes)

                inst_capsule_acts = capsule_acts[i, :, y_coords, x_coords]    # (n_caps, p)
                inst_capsule_acts = inst_capsule_acts.view(self.n_init_capsules[2]*len(y_coords), )  # (n_caps*p, )

                out_capsule_poses, out_capsule_acts = self.transformer_routing(inst_capsule_votes, inst_capsule_acts)  # (34, F_out), (34, )
                
                inst_points_down4 = torch.unique(inst_points // 4, dim=1)
                y_coords, x_coords = inst_points_down4[0, :], inst_points_down4[1, :]

                instance_poses[i, y_coords, x_coords] = out_capsule_poses.cpu()
                instance_acts[i, y_coords, x_coords] = out_capsule_acts.cpu()
                
                class_outs.append(out_capsule_acts)

            class_outputs.append(torch.stack(class_outs) if len(class_outs) != 0 else [])
            
        fg_preds, reg_preds, class_preds, inst_maps_preds, seg_list_preds = [], [], [], [], []
        
        fg_preds.append(fg_pred)
        reg_preds.append(regressions)
        class_preds.append([class8to34(class_output) for class_output in class_outputs])
        inst_maps_preds.append(inst_maps)
        seg_list_preds.append(segmentation_lists)
        
        if two_stage:
            new_capsules_poses = torch.cat((fgbg_poses.cuda(), instance_poses.cuda()), -2)
            new_capsules_acts = torch.cat((fgbg_acts.cuda(), instance_acts.cuda()), -1)

            # new_capsules_poses = new_capsules_poses.detach()
            # new_capsules_acts = new_capsules_acts.detach()

            # capsule_poses = capsule_poses.detach()
            # capsule_acts = capsule_acts.detach()


            new_capsules_poses = new_capsules_poses.view(b_size, h_new, w_new, -1).permute(0, 3, 1, 2)
            capsule_votes_seg2 = self.vote_transform_seg2(new_capsules_poses)  # (batch_size, n_caps*vote_dim, h/4, w/4)
            capsule_votes_seg3 = self.vote_transform_seg3(capsule_votes_seg2)
            capsule_votes_seg3 = capsule_votes_seg3.permute(0, 2, 3, 1).view(b_size, h_new, w_new, self.num_inst_classes+2, self.vote_dim_seg)

            fgbg_poses, fgbg_acts = self.transformer_routing_seg2(capsule_votes_seg3, new_capsules_acts)

            fg_pred = fgbg_acts[..., 0].unsqueeze(1)  # Shape (batch_size, 1, h/16, w/16)

            fg_poses = fgbg_poses[..., 0, :]
            regressions = self.regression_linear(fg_poses).permute(0, 3, 1, 2)
            regressions = F.tanh(regressions)

            # Resizes the output maps
            fg_pred = F.upsample(fg_pred, size=(h, w), mode="bilinear")  # (shape: (batch_size, num_classes, h, w))
            center = torch.zeros_like(fg_pred)  # F.upsample(center, size=(h, w), mode="bilinear")
            regressions = F.upsample(regressions, size=(h, w), mode="bilinear")
            regressions[:, 0] = regressions[:, 0] * w
            regressions[:, 1] = regressions[:, 1] * h

            capsule_votes_class = self.vote_transform_class2(capsule_poses)  # (batch_size, n_caps*vote_dim, h/16, w/16)

            if point_lists is None:
                # inst_maps, point_lists, segmentation_lists = self.hough_routing(fg_pred, regressions, gt_seg)
                if gt_reg is None:
                    inst_maps, point_lists2, segmentation_lists = self.hough_routing(fg_pred, regressions, gt_seg)
                else:
                    inst_maps, point_lists2, segmentation_lists = self.hough_routing(fg_pred, gt_reg, gt_seg)
            else:
                inst_maps = []
                segmentation_lists = []
                point_lists2 = point_lists

            class_outputs = []
            
            for i, point_list in enumerate(point_lists2):

                class_outs = []
                for inst_points in point_list:
                    # gather capsules corresponding to inst_points
                    inst_points = torch.unique(inst_points // 16, dim=1)

                    y_coords, x_coords = inst_points[0, :], inst_points[1, :]

                    inst_capsule_votes = capsule_votes_class[i, :, y_coords, x_coords]  # (n_caps*vote_dim, p)
                    inst_capsule_votes = inst_capsule_votes.view(self.n_init_capsules[2], self.vote_dim, len(y_coords))  # (n_caps, vote_dim, p)
                    inst_capsule_votes = torch.transpose(inst_capsule_votes, 1, 2).reshape(self.n_init_capsules[2]*len(y_coords), self.vote_dim)  # (n_caps*p, vote_dim)

                    #assert config.positional_encoding == False  # positional encoding will require more changes
                    if config.positional_encoding == True:
                        inst_points_mean = torch.mean(inst_points.float(), 0, keepdim=True)
                        inst_points_rel = inst_points - inst_points_mean  # gets the relative coordinates
                        y_coords_rel, x_coords_rel = inst_points_rel[0, :], inst_points_rel[1, :]
                        y_coords_rel, x_coords_rel = y_coords_rel / float(h / 16), x_coords_rel / float(w / 16)  # Performs normalization between 0 and 1

                        # x_coords_rel and y_coords_rel should be of shape (p, )
                        x_coords_rel = x_coords_rel.unsqueeze(0).repeat(self.n_init_capsules[2], 1).reshape(self.n_init_capsules[2]*len(y_coords), )  # makes x_coords_rel of shape (N*P, )
                        y_coords_rel = y_coords_rel.unsqueeze(0).repeat(self.n_init_capsules[2], 1).reshape(self.n_init_capsules[2]*len(y_coords), )  # makes y_coords_rel of shape (N*P, )

                        if config.positional_encoding_type == 'addition':
                            inst_capsule_votes[:, -1] += x_coords_rel.cuda()
                            inst_capsule_votes[:, -2] += y_coords_rel.cuda()
                        elif config.positional_encoding_type == 'concat':
                            inst_capsule_votes = torch.cat((inst_capsule_votes, y_coords_rel.unsqueeze(1).float().cuda(), x_coords_rel.unsqueeze(1).float().cuda()), 1)

                        inst_capsule_votes = self.pos_vote_transform2(inst_capsule_votes)

                    inst_capsule_acts = capsule_acts[i, :, y_coords, x_coords]    # (n_caps, p)
                    inst_capsule_acts = inst_capsule_acts.view(self.n_init_capsules[2]*len(y_coords), )  # (n_caps*p, )

                    out_capsule_poses, out_capsule_acts = self.transformer_routing2(inst_capsule_votes, inst_capsule_acts)  # (34, F_out), (34, )
                    
                    class_outs.append(out_capsule_acts)

                class_outputs.append(torch.stack(class_outs) if len(class_outs) != 0 else [])
                            
            fg_preds.append(fg_pred)
            reg_preds.append(regressions)
            class_preds.append([class8to34(class_output) for class_output in class_outputs])
            inst_maps_preds.append(inst_maps)
            seg_list_preds.append(segmentation_lists)
            

        # Should output center with shape (B, 1, H/16, W/16)
        # and regressions with shape(B, 2, H/16, W/16)
        return fg_preds, reg_preds, class_preds, inst_maps_preds, seg_list_preds

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)

class NewCapsuleModel6(nn.Module):
    def __init__(self, model_id, project_dir):
        super(NewCapsuleModel6, self).__init__()
        self.num_classes = config.n_classes
        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        self.resnet = ResNet50_OS16()  # NOTE! specify the type of ResNet here

        self.aspp = ASPP_Bottleneck()

        in_feats = 1280

        self.hough_routing = HoughRouting1()

        self.n_init_capsules = [4, 4, 8]
        self.init_capsule_dim = [4, 8, 32]
        
        self.num_inst_classes = 8

        self.primary_capsules_skip_4 = PrimaryCaps(256, self.n_init_capsules[0], 1, 1, self.init_capsule_dim[0])
        self.primary_capsules_skip_8 = PrimaryCaps(512, self.n_init_capsules[1], 1, 1, self.init_capsule_dim[1])
        self.primary_capsules = PrimaryCaps(in_feats, self.n_init_capsules[2], 1, 1, self.init_capsule_dim[2])

        self.vote_dim_seg = 32
        self.vote_transform_seg_skip_4 = VotingModule(self.n_init_capsules[0], self.init_capsule_dim[0], self.vote_dim_seg)
        self.vote_transform_seg_skip_8 = VotingModule(self.n_init_capsules[1], self.init_capsule_dim[1], self.vote_dim_seg)
        self.vote_transform_seg = VotingModule(self.n_init_capsules[2], self.init_capsule_dim[2], self.vote_dim_seg)

        self.vote_dim = config.vote_dim
        self.vote_transform_class = VotingModule(self.n_init_capsules[2], self.init_capsule_dim[2], self.vote_dim)

        if config.positional_encoding == True:
            if config.positional_encoding_type == 'addition':
                self.pos_vote_transform = nn.Linear(self.vote_dim, self.vote_dim)  # Will need to add two to first argument if concatenating positional encoding
            elif config.positional_encoding_type == 'concat':
                self.pos_vote_transform = nn.Linear(self.vote_dim + 2, self.vote_dim)  # Will need to add two to first argument if concatenating positional encoding
        self.noise_scale = 4.0

        self.transformer_routing = TransformerRouting(n_feats_in=self.vote_dim, n_caps_out=self.num_inst_classes, output_dim=16, use_vote_transform=False)

        self.transformer_routing_seg = TransformerRouting(n_feats_in=self.vote_dim_seg, n_caps_out=2, hidden_dim=self.vote_dim_seg, output_dim=16, use_vote_transform=False)
        self.regression_linear = nn.Linear(16, 2)
        
        self.vote_transform_seg2 = VotingModule(2 + self.num_inst_classes, 16, 16, kernel_dim=3, dilation=3)
        self.vote_transform_seg3 = VotingModule(2 + self.num_inst_classes, 16, self.vote_dim_seg, kernel_dim=3, dilation=3)

        self.transformer_routing_seg2 = TransformerRouting(n_feats_in=self.vote_dim_seg, n_caps_out=2, hidden_dim=self.vote_dim_seg, output_dim=16, use_vote_transform=False)
        self.regression_linear2 = nn.Linear(16, 2)

        self.vote_transform_class2 = VotingModule(self.n_init_capsules[2], self.init_capsule_dim[2], self.vote_dim)

        if config.positional_encoding == True:
            if config.positional_encoding_type == 'addition':
                self.pos_vote_transform2 = nn.Linear(self.vote_dim, self.vote_dim)  # Will need to add two to first argument if concatenating positional encoding
            elif config.positional_encoding_type == 'concat':
                self.pos_vote_transform2 = nn.Linear(self.vote_dim + 2, self.vote_dim)  # Will need to add two to first argument if concatenating positional encoding

        self.transformer_routing2 = TransformerRouting(n_feats_in=self.vote_dim, n_caps_out=self.num_inst_classes, output_dim=16, use_vote_transform=False)
        
    def get_primary_and_fg_capsules(self, feature_output, skip_8, skip_4, input_size):
        h, w = input_size
    
        # creates the first capsule layer
        capsule_poses_s4, capsule_acts_s4 = self.primary_capsules_skip_4(skip_4)
        capsule_poses_s8, capsule_acts_s8 = self.primary_capsules_skip_8(skip_8)
    
        capsule_poses, capsule_acts = self.primary_capsules(feature_output)

        capsule_votes_seg_s4 = self.vote_transform_seg_skip_4(capsule_poses_s4)  # (batch_size, n_caps*vote_dim, h/4, w/4)
        capsule_votes_seg_s8 = self.vote_transform_seg_skip_8(capsule_poses_s8)  # (batch_size, n_caps*vote_dim, h/8, w/8)
        capsule_votes_seg = self.vote_transform_seg(capsule_poses)  # (batch_size, n_caps*vote_dim, h/16, w/16)

        capsule_votes_seg_up_s8 = F.upsample(capsule_votes_seg_s8, size=(h//4, w//4), mode="bilinear")
        capsule_acts_up_s8 = F.upsample(capsule_acts_s8, size=(h//4, w//4), mode="bilinear")

        capsule_votes_seg_up = F.upsample(capsule_votes_seg, size=(h//4, w//4), mode="bilinear")
        capsule_acts_up = F.upsample(capsule_acts, size=(h//4, w//4), mode="bilinear")

        b_size, _, h_new, w_new = capsule_votes_seg_up.shape

        capsule_votes_seg_s4 = capsule_votes_seg_s4.permute(0, 2, 3, 1).view(b_size, h_new, w_new, self.n_init_capsules[0], self.vote_dim_seg) # (B, h', w', 4, 32)
        capsule_votes_seg_s8 = capsule_votes_seg_up_s8.permute(0, 2, 3, 1).view(b_size, h_new, w_new, self.n_init_capsules[1], self.vote_dim_seg) # (B, h', w', 8, 32)
        capsule_votes_seg = capsule_votes_seg_up.permute(0, 2, 3, 1).view(b_size, h_new, w_new, self.n_init_capsules[2], self.vote_dim_seg) # (B, h', w', 32, 32)
        capsule_votes_seg = torch.cat((capsule_votes_seg, capsule_votes_seg_s8, capsule_votes_seg_s4), -2) # (B, h, w, 16, 32)
        capsule_acts_seg = torch.cat((capsule_acts_up, capsule_acts_up_s8, capsule_acts_s4), 1).permute(0, 2, 3, 1) # (B, h, w, 16)

        fgbg_poses, fgbg_acts = self.transformer_routing_seg(capsule_votes_seg, capsule_acts_seg)

        return (capsule_poses, capsule_acts), (fgbg_poses, fgbg_acts)    

    def create_inst_maps(self, point_lists, gt_reg, gt_seg, fg_pred, regressions):
        if point_lists is None:
            # inst_maps, point_lists, segmentation_lists = self.hough_routing(fg_pred, regressions, gt_seg)
            if gt_reg is None:
                inst_maps, point_lists, segmentation_lists = self.hough_routing(fg_pred, regressions, gt_seg)
            else:
                inst_maps, point_lists, segmentation_lists = self.hough_routing(fg_pred, gt_reg, gt_seg)
        else:
            inst_maps = []
            segmentation_lists = []
            
        return point_lists, inst_maps, segmentation_lists
        
    def scatter_capsules(self, point_lists, capsule_votes_inst, capsule_acts, input_size, instance_scale):
        h, w = input_size
        b_size, _, h_inp, w_inp = capsule_votes_inst.shape
        assert h//h_inp == w/w_inp
        capsule_scale = h//h_inp
    
        instance_poses = torch.zeros((b_size, h//4, w//4, self.num_inst_classes, 16))
        instance_acts = torch.zeros((b_size, h//4, w//4, self.num_inst_classes))
        
        class_outputs = []
        for i, point_list in enumerate(point_lists):

            class_outs = []
            for inst_points in point_list:
                # gather capsules corresponding to inst_points
                inst_points = torch.unique(inst_points // 16, dim=1)

                y_coords, x_coords = inst_points[0, :], inst_points[1, :]

                inst_capsule_votes = capsule_votes_inst[i, :, y_coords, x_coords]  # (n_caps*vote_dim, p)
                inst_capsule_votes = inst_capsule_votes.view(self.n_init_capsules[2], self.vote_dim, len(y_coords))  # (n_caps, vote_dim, p)
                inst_capsule_votes = torch.transpose(inst_capsule_votes, 1, 2).reshape(self.n_init_capsules[2]*len(y_coords), self.vote_dim)  # (n_caps*p, vote_dim)

                #assert config.positional_encoding == False  # positional encoding will require more changes
                if config.positional_encoding == True:
                    inst_points_mean = torch.mean(inst_points.float(), 0, keepdim=True)
                    inst_points_rel = inst_points - inst_points_mean  # gets the relative coordinates
                    y_coords_rel, x_coords_rel = inst_points_rel[0, :], inst_points_rel[1, :]
                    y_coords_rel, x_coords_rel = y_coords_rel / float(h / 16), x_coords_rel / float(w / 16)  # Performs normalization between 0 and 1

                    # x_coords_rel and y_coords_rel should be of shape (p, )
                    x_coords_rel = x_coords_rel.unsqueeze(0).repeat(self.n_init_capsules[2], 1).reshape(self.n_init_capsules[2]*len(y_coords), )  # makes x_coords_rel of shape (N*P, )
                    y_coords_rel = y_coords_rel.unsqueeze(0).repeat(self.n_init_capsules[2], 1).reshape(self.n_init_capsules[2]*len(y_coords), )  # makes y_coords_rel of shape (N*P, )

                    if config.positional_encoding_type == 'addition':
                        inst_capsule_votes[:, -1] += x_coords_rel.cuda()
                        inst_capsule_votes[:, -2] += y_coords_rel.cuda()
                    elif config.positional_encoding_type == 'concat':
                        inst_capsule_votes = torch.cat((inst_capsule_votes, y_coords_rel.unsqueeze(1).float().cuda(), x_coords_rel.unsqueeze(1).float().cuda()), 1)

                    inst_capsule_votes = self.pos_vote_transform(inst_capsule_votes)

                inst_capsule_acts = capsule_acts[i, :, y_coords, x_coords]    # (n_caps, p)
                inst_capsule_acts = inst_capsule_acts.view(self.n_init_capsules[2]*len(y_coords), )  # (n_caps*p, )

                out_capsule_poses, out_capsule_acts = self.transformer_routing(inst_capsule_votes, inst_capsule_acts)  # (34, F_out), (34, )
                
                inst_points_down4 = torch.unique(inst_points // 4, dim=1)
                y_coords, x_coords = inst_points_down4[0, :], inst_points_down4[1, :]

                instance_poses[i, y_coords, x_coords] = out_capsule_poses.cpu()
                instance_acts[i, y_coords, x_coords] = out_capsule_acts.cpu()
                
                class_outs.append(out_capsule_acts)

            class_outputs.append(torch.stack(class_outs) if len(class_outs) != 0 else [])
            
        return instance_poses, instance_acts, class_outputs
        
    def scatter_capsules2(self, point_lists, capsule_votes_inst, capsule_acts, input_size):
        h, w = input_size
        b_size, _, h_inp, w_inp = capsule_votes_inst.shape
        assert h//h_inp == w/w_inp
        capsule_scale = h//h_inp

        class_outputs = []
        for i, point_list in enumerate(point_lists):

            class_outs = []
            for inst_points in point_list:
                # gather capsules corresponding to inst_points
                inst_points_down16 = torch.unique(inst_points // capsule_scale, dim=1)

                y_coords, x_coords = inst_points_down16[0, :], inst_points_down16[1, :]

                inst_capsule_votes = capsule_votes_inst[i, :, y_coords, x_coords]  # (n_caps*vote_dim, p)
                inst_capsule_votes = inst_capsule_votes.view(self.n_init_capsules[2], self.vote_dim, len(y_coords))  # (n_caps, vote_dim, p)
                inst_capsule_votes = torch.transpose(inst_capsule_votes, 1, 2).reshape(self.n_init_capsules[2] * len(y_coords), self.vote_dim)  # (n_caps*p, vote_dim)

                if config.positional_encoding == True:
                    inst_points_mean = torch.mean(inst_points.float(), 0, keepdim=True)
                    inst_points_rel = inst_points - inst_points_mean  # gets the relative coordinates
                    y_coords_rel, x_coords_rel = inst_points_rel[0, :], inst_points_rel[1, :]
                    y_coords_rel, x_coords_rel = y_coords_rel / float(h / capsule_scale), x_coords_rel / float(w / capsule_scale)  # Performs normalization between 0 and 1

                    # x_coords_rel and y_coords_rel should be of shape (p, )
                    x_coords_rel = x_coords_rel.unsqueeze(0).repeat(self.n_init_capsules[2], 1).reshape(self.n_init_capsules[2] * len(y_coords), )  # makes x_coords_rel of shape (N*P, )
                    y_coords_rel = y_coords_rel.unsqueeze(0).repeat(self.n_init_capsules[2], 1).reshape(self.n_init_capsules[2] * len(y_coords), )  # makes y_coords_rel of shape (N*P, )

                    if config.positional_encoding_type == 'addition':
                        inst_capsule_votes[:, -1] += x_coords_rel.cuda()
                        inst_capsule_votes[:, -2] += y_coords_rel.cuda()
                    elif config.positional_encoding_type == 'concat':
                        inst_capsule_votes = torch.cat((inst_capsule_votes, y_coords_rel.unsqueeze(1).float().cuda(), x_coords_rel.unsqueeze(1).float().cuda()), 1)

                    inst_capsule_votes = self.pos_vote_transform2(inst_capsule_votes)

                inst_capsule_acts = capsule_acts[i, :, y_coords, x_coords]  # (n_caps, p)
                inst_capsule_acts = inst_capsule_acts.view(self.n_init_capsules[2] * len(y_coords), )  # (n_caps*p, )

                out_capsule_poses, out_capsule_acts = self.transformer_routing2(inst_capsule_votes, inst_capsule_acts)  # (34, F_out), (34, )

                class_outs.append(out_capsule_acts)

            class_outputs.append(torch.stack(class_outs) if len(class_outs) != 0 else [])

        return class_outputs

        
    def forward(self, x, point_lists=None, gt_seg=None, gt_reg=None, two_stage=False):
        # (x has shape (batch_size, 3, h, w))
        h = x.size()[2]
        w = x.size()[3]

        # Encoder:
        feature_map, skip_8, skip_4 = self.resnet(x)  # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNe$

        feature_output = self.aspp(feature_map)  # (shape: (batch_size, 1280, h/16, w/16))

        (primary_poses, primary_acts), (fgbg_poses, fgbg_acts) = self.get_primary_and_fg_capsules(feature_output, skip_8, skip_4, (h, w))

        fg_pred = fgbg_acts[..., 0].unsqueeze(1)  # Shape (batch_size, 1, h/16, w/16)

        fg_poses = fgbg_poses[..., 0, :]
        regressions = self.regression_linear(fg_poses).permute(0, 3, 1, 2)
        regressions = F.tanh(regressions)

        # Resizes the output maps
        fg_pred = F.upsample(fg_pred, size=(h, w), mode="bilinear")  # (shape: (batch_size, num_classes, h, w))
        center = torch.zeros_like(fg_pred)  # F.upsample(center, size=(h, w), mode="bilinear")
        regressions = F.upsample(regressions, size=(h, w), mode="bilinear")
        regressions[:, 0] = regressions[:, 0] * w
        regressions[:, 1] = regressions[:, 1] * h

        #capsule_votes_class = self.vote_transform_class(primary_poses)  # (batch_size, n_caps*vote_dim, h/16, w/16)

        point_lists1, inst_maps, segmentation_lists = self.create_inst_maps(point_lists, gt_reg, gt_seg, fg_pred, regressions)

        class_outputs = []
        
        capsule_votes_inst = self.vote_transform_class(primary_poses)  # (batch_size, n_caps*vote_dim, h/16, w/16)
        
        instance_poses, instance_acts, class_outputs = self.scatter_capsules(point_lists1, capsule_votes_inst, primary_acts, (h, w), instance_scale=4)
            
        fg_preds, reg_preds, class_preds, inst_maps_preds, seg_list_preds = [], [], [], [], []
        
        fg_preds.append(fg_pred)
        reg_preds.append(regressions)
        class_preds.append([class8to34(class_output) for class_output in class_outputs])
        inst_maps_preds.append(inst_maps)
        seg_list_preds.append(segmentation_lists)
        
        if two_stage:
            new_capsules_poses = torch.cat((fgbg_poses.cuda(), instance_poses.cuda()), -2)
            new_capsules_acts = torch.cat((fgbg_acts.cuda(), instance_acts.cuda()), -1)

            if config.stop_grad:
                new_capsules_poses = new_capsules_poses.detach()
                new_capsules_acts = new_capsules_acts.detach()

                capsule_poses = primary_poses.detach()
                capsule_acts = primary_acts.detach()
            else:
                capsule_poses = primary_poses
                capsule_acts = primary_acts

            b_size, h_new, w_new, _, _ = fgbg_poses.shape

            new_capsules_poses = new_capsules_poses.view(b_size, h_new, w_new, -1).permute(0, 3, 1, 2)
            capsule_votes_seg2 = self.vote_transform_seg2(new_capsules_poses)  # (batch_size, n_caps*vote_dim, h/4, w/4)
            capsule_votes_seg3 = self.vote_transform_seg3(capsule_votes_seg2)
            capsule_votes_seg3 = capsule_votes_seg3.permute(0, 2, 3, 1).view(b_size, h_new, w_new, self.num_inst_classes+2, self.vote_dim_seg)

            fgbg_poses, fgbg_acts = self.transformer_routing_seg2(capsule_votes_seg3, new_capsules_acts)

            fg_pred = fgbg_acts[..., 0].unsqueeze(1)  # Shape (batch_size, 1, h/16, w/16)

            fg_poses = fgbg_poses[..., 0, :]
            regressions = self.regression_linear2(fg_poses).permute(0, 3, 1, 2)
            regressions = F.tanh(regressions)

            # Resizes the output maps
            fg_pred = F.upsample(fg_pred, size=(h, w), mode="bilinear")  # (shape: (batch_size, num_classes, h, w))
            center = torch.zeros_like(fg_pred)  # F.upsample(center, size=(h, w), mode="bilinear")
            regressions = F.upsample(regressions, size=(h, w), mode="bilinear")
            regressions[:, 0] = regressions[:, 0] * w
            regressions[:, 1] = regressions[:, 1] * h

            capsule_votes_inst = self.vote_transform_class2(capsule_poses)  # (batch_size, n_caps*vote_dim, h/16, w/16)

            point_lists2, inst_maps, segmentation_lists = self.create_inst_maps(point_lists, gt_reg, gt_seg, fg_pred, regressions)
            
            class_outputs = self.scatter_capsules2(point_lists2, capsule_votes_inst, capsule_acts, (h, w))
                            
            fg_preds.append(fg_pred)
            reg_preds.append(regressions)
            class_preds.append([class8to34(class_output) for class_output in class_outputs])
            inst_maps_preds.append(inst_maps)
            seg_list_preds.append(segmentation_lists)
            

        # Should output center with shape (B, 1, H/16, W/16)
        # and regressions with shape(B, 2, H/16, W/16)
        return fg_preds, reg_preds, class_preds, inst_maps_preds, seg_list_preds

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)


class CapsuleModel7(nn.Module):
    def __init__(self, model_id, project_dir):
        super(CapsuleModel7, self).__init__()
        self.num_classes = config.n_classes
        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        self.resnet = ResNet50_OS16()  # NOTE! specify the type of ResNet here

        self.aspp = ASPP_Bottleneck()

        in_feats = 1280

        self.hough_routing = HoughRouting1()

        self.n_init_capsules = [4, 4, 8]
        self.init_capsule_dim = [4, 8, 32]
        
        self.num_inst_classes = 8

        self.primary_capsules_skip_4 = PrimaryCaps(256, self.n_init_capsules[0], 1, 1, self.init_capsule_dim[0])
        self.primary_capsules_skip_8 = PrimaryCaps(512, self.n_init_capsules[1], 1, 1, self.init_capsule_dim[1])
        self.primary_capsules = PrimaryCaps(in_feats, self.n_init_capsules[2], 1, 1, self.init_capsule_dim[2])

        self.vote_dim_seg = 32
        self.vote_transform_seg_skip_4 = VotingModule(self.n_init_capsules[0], self.init_capsule_dim[0], self.vote_dim_seg)
        self.vote_transform_seg_skip_8 = VotingModule(self.n_init_capsules[1], self.init_capsule_dim[1], self.vote_dim_seg)
        self.vote_transform_seg = VotingModule(self.n_init_capsules[2], self.init_capsule_dim[2], self.vote_dim_seg)

        self.vote_dim = config.vote_dim
        self.vote_transform_class = VotingModule(self.n_init_capsules[2], self.init_capsule_dim[2], self.vote_dim)
        self.vote_transform_class_dense = VotingModule(self.n_init_capsules[2], self.init_capsule_dim[2], self.vote_dim)

        if config.positional_encoding == True:
            if config.positional_encoding_type == 'addition':
                self.pos_vote_transform = nn.Linear(self.vote_dim, self.vote_dim)  # Will need to add two to first argument if concatenating positional encoding
            elif config.positional_encoding_type == 'concat':
                self.pos_vote_transform = nn.Linear(self.vote_dim + 2, self.vote_dim)  # Will need to add two to first argument if concatenating positional encoding
        self.noise_scale = 4.0

        self.transformer_routing = TransformerRouting(n_feats_in=self.vote_dim, n_caps_out=self.num_inst_classes, output_dim=16, use_vote_transform=False)
        self.transformer_routing_dense = TransformerRouting(n_feats_in=self.vote_dim, n_caps_out=self.num_inst_classes, output_dim=16, use_vote_transform=False)

        self.transformer_routing_seg = TransformerRouting(n_feats_in=self.vote_dim_seg, n_caps_out=2, hidden_dim=self.vote_dim_seg, output_dim=16, use_vote_transform=False)
        self.regression_linear = nn.Linear(16, 2)
        
        self.vote_transform_seg2 = VotingModule(2 + self.num_inst_classes, 16, 16, kernel_dim=3, dilation=3)
        self.vote_transform_seg3 = VotingModule(2 + self.num_inst_classes, 16, self.vote_dim_seg, kernel_dim=3, dilation=3)

        self.transformer_routing_seg2 = TransformerRouting(n_feats_in=self.vote_dim_seg, n_caps_out=2, hidden_dim=self.vote_dim_seg, output_dim=16, use_vote_transform=False)
        self.regression_linear2 = nn.Linear(16, 2)

        self.vote_transform_class2 = VotingModule(self.n_init_capsules[2], self.init_capsule_dim[2], self.vote_dim)

        if config.positional_encoding == True:
            if config.positional_encoding_type == 'addition':
                self.pos_vote_transform2 = nn.Linear(self.vote_dim, self.vote_dim)  # Will need to add two to first argument if concatenating positional encoding
            elif config.positional_encoding_type == 'concat':
                self.pos_vote_transform2 = nn.Linear(self.vote_dim + 2, self.vote_dim)  # Will need to add two to first argument if concatenating positional encoding

        self.transformer_routing2 = TransformerRouting(n_feats_in=self.vote_dim, n_caps_out=self.num_inst_classes, output_dim=16, use_vote_transform=False)
        
    def get_primary_and_fg_capsules(self, feature_output, skip_8, skip_4, input_size):
        h, w = input_size

        # creates the first capsule layer
        capsule_poses_s4, capsule_acts_s4 = self.primary_capsules_skip_4(skip_4)
        capsule_poses_s8, capsule_acts_s8 = self.primary_capsules_skip_8(skip_8)

        capsule_poses, capsule_acts = self.primary_capsules(feature_output)

        capsule_votes_seg_s4 = self.vote_transform_seg_skip_4(capsule_poses_s4)  # (batch_size, n_caps*vote_dim, h/4, w/4)
        capsule_votes_seg_s8 = self.vote_transform_seg_skip_8(capsule_poses_s8)  # (batch_size, n_caps*vote_dim, h/8, w/8)
        capsule_votes_seg = self.vote_transform_seg(capsule_poses)  # (batch_size, n_caps*vote_dim, h/16, w/16)

        capsule_votes_seg_up_s8 = F.upsample(capsule_votes_seg_s8, size=(h//4, w//4), mode="bilinear")
        capsule_acts_up_s8 = F.upsample(capsule_acts_s8, size=(h//4, w//4), mode="bilinear")

        capsule_votes_seg_up = F.upsample(capsule_votes_seg, size=(h//4, w//4), mode="bilinear")
        capsule_acts_up = F.upsample(capsule_acts, size=(h//4, w//4), mode="bilinear")

        b_size, _, h_new, w_new = capsule_votes_seg_up.shape

        capsule_votes_seg_s4 = capsule_votes_seg_s4.permute(0, 2, 3, 1).view(b_size, h_new, w_new, self.n_init_capsules[0], self.vote_dim_seg) # (B, h', w', 4, 32)
        capsule_votes_seg_s8 = capsule_votes_seg_up_s8.permute(0, 2, 3, 1).view(b_size, h_new, w_new, self.n_init_capsules[1], self.vote_dim_seg) # (B, h', w', 8, 32)
        capsule_votes_seg = capsule_votes_seg_up.permute(0, 2, 3, 1).view(b_size, h_new, w_new, self.n_init_capsules[2], self.vote_dim_seg) # (B, h', w', 32, 32)
        capsule_votes_seg = torch.cat((capsule_votes_seg, capsule_votes_seg_s8, capsule_votes_seg_s4), -2) # (B, h, w, 16, 32)
        capsule_acts_seg = torch.cat((capsule_acts_up, capsule_acts_up_s8, capsule_acts_s4), 1).permute(0, 2, 3, 1) # (B, h, w, 16)

        fgbg_poses, fgbg_acts = self.transformer_routing_seg(capsule_votes_seg, capsule_acts_seg)

        return (capsule_poses, capsule_acts), (fgbg_poses, fgbg_acts)

    def create_inst_maps(self, point_lists, gt_reg, gt_seg, fg_pred, regressions):
        if point_lists is None:
            # inst_maps, point_lists, segmentation_lists = self.hough_routing(fg_pred, regressions, gt_seg)
            if gt_reg is None:
                inst_maps, point_lists, segmentation_lists = self.hough_routing(fg_pred, regressions, gt_seg)
            else:
                inst_maps, point_lists, segmentation_lists = self.hough_routing(fg_pred, gt_reg, gt_seg)
        else:
            inst_maps = []
            segmentation_lists = []
            
        return point_lists, inst_maps, segmentation_lists
        
    def scatter_capsules(self, point_lists, capsule_votes_inst, capsule_acts, input_size, instance_scale):
        h, w = input_size
        b_size, _, h_inp, w_inp = capsule_votes_inst.shape
        assert h//h_inp == w/w_inp
        capsule_scale = h//h_inp

        instance_poses = torch.zeros((b_size, h//4, w//4, self.num_inst_classes, 16))
        instance_acts = torch.zeros((b_size, h//4, w//4, self.num_inst_classes))
        
        class_outputs = []
        for i, point_list in enumerate(point_lists):

            class_outs = []
            for inst_points in point_list:
                # gather capsules corresponding to inst_points
                inst_points = torch.unique(inst_points // 16, dim=1)

                y_coords, x_coords = inst_points[0, :], inst_points[1, :]

                inst_capsule_votes = capsule_votes_inst[i, :, y_coords, x_coords]  # (n_caps*vote_dim, p)
                inst_capsule_votes = inst_capsule_votes.view(self.n_init_capsules[2], self.vote_dim, len(y_coords))  # (n_caps, vote_dim, p)
                inst_capsule_votes = torch.transpose(inst_capsule_votes, 1, 2).reshape(self.n_init_capsules[2]*len(y_coords), self.vote_dim)  # (n_caps*p, vote_dim)

                #assert config.positional_encoding == False  # positional encoding will require more changes
                if config.positional_encoding == True:
                    inst_points_mean = torch.mean(inst_points.float(), 0, keepdim=True)
                    inst_points_rel = inst_points - inst_points_mean  # gets the relative coordinates
                    y_coords_rel, x_coords_rel = inst_points_rel[0, :], inst_points_rel[1, :]
                    y_coords_rel, x_coords_rel = y_coords_rel / float(h / 16), x_coords_rel / float(w / 16)  # Performs normalization between 0 and 1

                    # x_coords_rel and y_coords_rel should be of shape (p, )
                    x_coords_rel = x_coords_rel.unsqueeze(0).repeat(self.n_init_capsules[2], 1).reshape(self.n_init_capsules[2]*len(y_coords), )  # makes x_coords_rel of shape (N*P, )
                    y_coords_rel = y_coords_rel.unsqueeze(0).repeat(self.n_init_capsules[2], 1).reshape(self.n_init_capsules[2]*len(y_coords), )  # makes y_coords_rel of shape (N*P, )

                    if config.positional_encoding_type == 'addition':
                        inst_capsule_votes[:, -1] += x_coords_rel.cuda()
                        inst_capsule_votes[:, -2] += y_coords_rel.cuda()
                    elif config.positional_encoding_type == 'concat':
                        inst_capsule_votes = torch.cat((inst_capsule_votes, y_coords_rel.unsqueeze(1).float().cuda(), x_coords_rel.unsqueeze(1).float().cuda()), 1)

                    inst_capsule_votes = self.pos_vote_transform(inst_capsule_votes)

                inst_capsule_acts = capsule_acts[i, :, y_coords, x_coords]    # (n_caps, p)
                inst_capsule_acts = inst_capsule_acts.view(self.n_init_capsules[2]*len(y_coords), )  # (n_caps*p, )

                out_capsule_poses, out_capsule_acts = self.transformer_routing(inst_capsule_votes, inst_capsule_acts)  # (34, F_out), (34, )
                
                inst_points_down4 = torch.unique(inst_points // 4, dim=1)
                y_coords, x_coords = inst_points_down4[0, :], inst_points_down4[1, :]

                instance_poses[i, y_coords, x_coords] = out_capsule_poses.cpu()
                instance_acts[i, y_coords, x_coords] = out_capsule_acts.cpu()
                
                class_outs.append(out_capsule_acts)

            class_outputs.append(torch.stack(class_outs) if len(class_outs) != 0 else [])
            
        return instance_poses, instance_acts, class_outputs
        
    def scatter_capsules2(self, point_lists, capsule_votes_inst, capsule_acts, input_size):
        h, w = input_size
        b_size, _, h_inp, w_inp = capsule_votes_inst.shape
        assert h//h_inp == w/w_inp
        capsule_scale = h//h_inp

        class_outputs = []
        for i, point_list in enumerate(point_lists):

            class_outs = []
            for inst_points in point_list:
                # gather capsules corresponding to inst_points
                inst_points_down16 = torch.unique(inst_points // capsule_scale, dim=1)

                y_coords, x_coords = inst_points_down16[0, :], inst_points_down16[1, :]

                inst_capsule_votes = capsule_votes_inst[i, :, y_coords, x_coords]  # (n_caps*vote_dim, p)
                inst_capsule_votes = inst_capsule_votes.view(self.n_init_capsules[2], self.vote_dim, len(y_coords))  # (n_caps, vote_dim, p)
                inst_capsule_votes = torch.transpose(inst_capsule_votes, 1, 2).reshape(self.n_init_capsules[2] * len(y_coords), self.vote_dim)  # (n_caps*p, vote_dim)

                if config.positional_encoding == True:
                    inst_points_mean = torch.mean(inst_points.float(), 0, keepdim=True)
                    inst_points_rel = inst_points - inst_points_mean  # gets the relative coordinates
                    y_coords_rel, x_coords_rel = inst_points_rel[0, :], inst_points_rel[1, :]
                    y_coords_rel, x_coords_rel = y_coords_rel / float(h / capsule_scale), x_coords_rel / float(w / capsule_scale)  # Performs normalization between 0 and 1

                    # x_coords_rel and y_coords_rel should be of shape (p, )
                    x_coords_rel = x_coords_rel.unsqueeze(0).repeat(self.n_init_capsules[2], 1).reshape(self.n_init_capsules[2] * len(y_coords), )  # makes x_coords_rel of shape (N*P, )
                    y_coords_rel = y_coords_rel.unsqueeze(0).repeat(self.n_init_capsules[2], 1).reshape(self.n_init_capsules[2] * len(y_coords), )  # makes y_coords_rel of shape (N*P, )

                    if config.positional_encoding_type == 'addition':
                        inst_capsule_votes[:, -1] += x_coords_rel.cuda()
                        inst_capsule_votes[:, -2] += y_coords_rel.cuda()
                    elif config.positional_encoding_type == 'concat':
                        inst_capsule_votes = torch.cat((inst_capsule_votes, y_coords_rel.unsqueeze(1).float().cuda(), x_coords_rel.unsqueeze(1).float().cuda()), 1)

                    inst_capsule_votes = self.pos_vote_transform2(inst_capsule_votes)

                inst_capsule_acts = capsule_acts[i, :, y_coords, x_coords]  # (n_caps, p)
                inst_capsule_acts = inst_capsule_acts.view(self.n_init_capsules[2] * len(y_coords), )  # (n_caps*p, )

                out_capsule_poses, out_capsule_acts = self.transformer_routing2(inst_capsule_votes, inst_capsule_acts)  # (34, F_out), (34, )

                class_outs.append(out_capsule_acts)

            class_outputs.append(torch.stack(class_outs) if len(class_outs) != 0 else [])

        return class_outputs

        
    def forward(self, x, point_lists=None, gt_seg=None, gt_reg=None, two_stage=False):
        # (x has shape (batch_size, 3, h, w))
        h = x.size()[2]
        w = x.size()[3]

        # Encoder:
        feature_map, skip_8, skip_4 = self.resnet(x)  # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNe$

        feature_output = self.aspp(feature_map)  # (shape: (batch_size, 1280, h/16, w/16))

        (primary_poses, primary_acts), (fgbg_poses, fgbg_acts) = self.get_primary_and_fg_capsules(feature_output, skip_8, skip_4, (h, w))

        fg_pred = fgbg_acts[..., 0].unsqueeze(1)  # Shape (batch_size, 1, h/16, w/16)

        fg_poses = fgbg_poses[..., 0, :]
        regressions = self.regression_linear(fg_poses).permute(0, 3, 1, 2)
        regressions = F.tanh(regressions)

        # Resizes the output maps
        fg_pred = F.upsample(fg_pred, size=(h, w), mode="bilinear")  # (shape: (batch_size, num_classes, h, w))
        center = torch.zeros_like(fg_pred)  # F.upsample(center, size=(h, w), mode="bilinear")
        regressions = F.upsample(regressions, size=(h, w), mode="bilinear")
        regressions[:, 0] = regressions[:, 0] * w
        regressions[:, 1] = regressions[:, 1] * h

        point_lists1, inst_maps, segmentation_lists = self.create_inst_maps(point_lists, gt_reg, gt_seg, fg_pred, regressions)

        class_outputs = []
        
        capsule_votes_inst = self.vote_transform_class(primary_poses)  # (batch_size, n_caps*vote_dim, h/16, w/16)
        
        instance_poses, instance_acts, class_outputs = self.scatter_capsules(point_lists1, capsule_votes_inst, primary_acts, (h, w), instance_scale=4)
        
        class_votes_dense = self.vote_transform_class_dense(primary_poses)
        b_size, _, h_new, w_new = class_votes_dense.shape
        class_votes_dense = class_votes_dense.permute(0, 2, 3, 1).view(b_size, h_new, w_new, self.n_init_capsules[2], self.vote_dim) # (B, h', w', 32, 32)
        dense_class_capsules_poses, dense_class_capsules_acts = self.transformer_routing_dense(class_votes_dense, primary_acts.permute(0, 2, 3, 1))
        
        class_outputs_dense = []
        for i, point_list in enumerate(point_lists1):
            class_outs_dense = []
            for inst_points in point_list:
                # gather capsules corresponding to inst_points
                inst_points = torch.unique(inst_points // 16, dim=1)

                y_coords, x_coords = inst_points[0, :], inst_points[1, :]
                
                inst_capsule_dense = dense_class_capsules_acts[i, y_coords, x_coords]  # (n, 8)
                
                class_outs_dense.append(class8to34(inst_capsule_dense))
                
            class_outputs_dense.append(class_outs_dense)
            
        fg_preds, reg_preds, class_preds, inst_maps_preds, seg_list_preds = [], [], [], [], []
        
        fg_preds.append(fg_pred)
        reg_preds.append(regressions)
        class_preds.append([class8to34(class_output) for class_output in class_outputs])
        inst_maps_preds.append(inst_maps)
        seg_list_preds.append(segmentation_lists)
        
        if two_stage:
            new_capsules_poses = torch.cat((fgbg_poses.cuda(), instance_poses.cuda()), -2)
            new_capsules_acts = torch.cat((fgbg_acts.cuda(), instance_acts.cuda()), -1)

            new_capsules_poses = new_capsules_poses.detach()
            new_capsules_acts = new_capsules_acts.detach()

            capsule_poses = primary_poses.detach()
            capsule_acts = primary_acts.detach()

            b_size, h_new, w_new, _, _ = fgbg_poses.shape

            new_capsules_poses = new_capsules_poses.view(b_size, h_new, w_new, -1).permute(0, 3, 1, 2)
            capsule_votes_seg2 = self.vote_transform_seg2(new_capsules_poses)  # (batch_size, n_caps*vote_dim, h/4, w/4)
            capsule_votes_seg3 = self.vote_transform_seg3(capsule_votes_seg2)
            capsule_votes_seg3 = capsule_votes_seg3.permute(0, 2, 3, 1).view(b_size, h_new, w_new, self.num_inst_classes+2, self.vote_dim_seg)

            fgbg_poses, fgbg_acts = self.transformer_routing_seg2(capsule_votes_seg3, new_capsules_acts)

            fg_pred = fgbg_acts[..., 0].unsqueeze(1)  # Shape (batch_size, 1, h/16, w/16)

            fg_poses = fgbg_poses[..., 0, :]
            regressions = self.regression_linear2(fg_poses).permute(0, 3, 1, 2)
            regressions = F.tanh(regressions)

            # Resizes the output maps
            fg_pred = F.upsample(fg_pred, size=(h, w), mode="bilinear")  # (shape: (batch_size, num_classes, h, w))
            center = torch.zeros_like(fg_pred)  # F.upsample(center, size=(h, w), mode="bilinear")
            regressions = F.upsample(regressions, size=(h, w), mode="bilinear")
            regressions[:, 0] = regressions[:, 0] * w
            regressions[:, 1] = regressions[:, 1] * h

            # capsule_votes_class = self.vote_transform_class2(capsule_poses)  # (batch_size, n_caps*vote_dim, h/16, w/16)
            capsule_votes_inst = self.vote_transform_class2(capsule_poses)  # (batch_size, n_caps*vote_dim, h/16, w/16)

            point_lists2, inst_maps, segmentation_lists = self.create_inst_maps(point_lists, gt_reg, gt_seg, fg_pred, regressions)
            
            class_outputs = self.scatter_capsules2(point_lists2, capsule_votes_inst, capsule_acts, (h, w))
                            
            fg_preds.append(fg_pred)
            reg_preds.append(regressions)
            class_preds.append([class8to34(class_output) for class_output in class_outputs])
            inst_maps_preds.append(inst_maps)
            seg_list_preds.append(segmentation_lists)
            

        # Should output center with shape (B, 1, H/16, W/16)
        # and regressions with shape(B, 2, H/16, W/16)
        return fg_preds, reg_preds, class_preds, inst_maps_preds, seg_list_preds, [class_outputs_dense]

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)
