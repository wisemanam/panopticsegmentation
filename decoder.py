import torch
import torch.nn as nn
import torch.nn.functional as F


class seg_decoder(nn.Module):
    def __init__(self, num_classes):
        super(seg_decoder, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(256, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_1x1_2 = nn.Conv2d(512, 64, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        self.conv_1x1_3 = nn.Conv2d(256, 32, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)
        
        self.conv_5x5_1 = nn.Conv2d(256+64, 256, kernel_size=5)
        self.bn_conv_5x5_1 = nn.BatchNorm2d(256)

        self.conv_5x5_2 = nn.Conv2d(256+32, 256, kernel_size=5)
        self.bn_conv_5x5_2 = nn.BatchNorm2d(256)

        self.conv_5x5_3 = nn.Conv2d(256, 256, kernel_size=5)
        self.bn_conv_5x5_3 = nn.BatchNorm2d(256)

        self.conv_1x1_4	= nn.Conv2d(256, 34, kernel_size=1)
       	self.bn_conv_1x1_4 = nn.BatchNorm2d(256)

    def forward(self, feature_map, skip_8, skip_4):
        # (feature_map has shape (batch_size, 256, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16.)

        feature_map_h = feature_map.size()[2]  # (== h/16)
        feature_map_w = feature_map.size()[3]  # (== w/16)

        out = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))  # (shape: (batch_size, num_classes, h/16, w/16))
        out = F.upsample(out, size=(feature_map_h*2, feature_map_w*2), mode="bilinear") # (shape: (batch_size, num_classes, h/8, w/8))
        
        # Should be skip connection:
        skip_8 = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(skip_8))) # (shape: (batch_size, num_classes, h/8, w/8))
        out = torch.cat((out, skip_8), 1)

        out = F.relu(self.bn_conv_5x5_1(self.conv_5x5_1(out))) # (shape: (batch_size, num_classes, h/8, w/8))
        out = F.upsample(out, size=(feature_map_h*4, feature_map_w*4), mode="bilinear") # (shape: (batch_size, num_classes, h/4, w/4))
        
        # Should be skip connection:
        skip_4 = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(skip_4))) # (shape: (batch_size, num_classes, h/4, w/4))
        out = torch.cat((out, skip_4), 1)

        out = F.relu(self.bn_conv_5x5_2(self.conv_5x5_2(out))) # (shape: (batch_size, num_classes, h/4, w/4))

        # Prediction:
        out = F.relu(self.bn_conv_5x5_3(self.conv_5x5_3(out))) # (shape: (batch_size, num_classes, h/4, w/4))
        out = F.relu(self.bn_conv_1x1_4(self.conv_1x1_4(out))) # (shape: (batch_size, num_classes, h/4, w/4))
        return out

class inst_decoder(nn.Module):
    def __init__(self, num_classes):
        super(inst_decoder, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(256, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_1x1_2 = nn.Conv2d(512, 64, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        self.conv_1x1_3 = nn.Conv2d(256, 32, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)
        
        self.conv_5x5_1 = nn.Conv2d(256+32, 128, kernel_size=5)
        self.bn_conv_5x5_1 = nn.BatchNorm2d(256)

        self.conv_5x5_2 = nn.Conv2d(128+16, 128, kernel_size=5)
        self.bn_conv_5x5_2 = nn.BatchNorm2d(256)
        
        self.conv_5x5_3 = nn.Conv2d(128, 32, kernel_size=5)
        self.bn_conv_5x5_3 = nn.BatchNorm2d(256)

        self.conv_1x1_4 = nn.Conv2d(32, 1, kernel_size=1)
        self.bn_conv_1x1_4 = nn.BatchNorm2d(256)

        self.conv_1x1_5 = nn.Conv2d(32, 2, kernel_size=1)
        self.bn_conv_1x1_5 = nn.BatchNorm2d(256)

    def forward(self, feature_map, skip_8, skip_4):
        # (x has shape (batch_size, 256, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16.)

        feature_map_h = feature_map.size()[2]  # (== h/16)
        feature_map_w = feature_map.size()[3]  # (== w/16)

        out = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))  # (shape: (batch_size, num_classes, h/16, w/16))
        out = F.upsample(out, size=(feature_map_h*2, feature_map_w*2), mode="bilinear") # (shape: (batch_size, num_classes, h/8, w/8))
        
        skip_8 = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(skip_8))) # (shape: (batch_size, num_classes, h/8, w/8))
        out = torch.cat((out, skip_8), 1)

        out = F.relu(self.bn_conv_5x5_1(self.conv_1x1_2(out))) # (shape: (batch_size, num_classes, h/8, w/8))
        out = F.upsample(out, size=(feature_map_h*4, feature_map_w*4), mode="bilinear") # (shape: (batch_size, num_classes, h/4, w/4))
        
        skip_4 = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(skip_4))) # (shape: (batch_size, num_classes, h/4, w/4))
        out = torch.cat((out, skip_4), 1)
        
        out = F.relu(self.bn_conv_5x5_2(self.conv_5x5_2(out))) # (shape: (batch_size, num_classes, h/4, w/4))
        
        # Prediction:
        out_center = F.relu(self.bn_conv_5x5_3(self.conv_5x5_3(out)))
        out_center = F.relu(self.bn_conv_1x1_4(self.conv_1x1_4(out)))

        out_regression = F.relu(self.bn_conv_5x5_3(self.conv_5x5_3(out)))
        out_regression = F.relu(self.bn_conv_1x1_5(self.conv_1x1_5(out)))

        return out_center, out_regression

