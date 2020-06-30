import torch
import torch.nn as nn
import torch.nn.functional as F


class seg_decoder(nn.Module):
    def __init__(self, num_classes):
        super(seg_decoder, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(num_classes, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_1x1_2 = nn.Conv2d(256, 64, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        self.conv_1x1_3 = nn.Conv2d(256, 32, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)
        
        self.conv_5x5_1 = nn.Conv2d(64, 256, kernel_size=1)
        self.bn_conv_5x5_1 = nn.BatchNorm2d(256)

        self.conv_5x5_2 = nn.Conv2d(32, 256, kernel_size=1)
        self.bn_conv_5x5_2 = nn.BatchNorm2d(256)
        
    def forward(self, x):
        # (x has shape (batch_size, num_classes, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16.)
        # x is the output of aspp(feature_map)

        x_h = x.size()[2]  # (== h/16)
        x_w = x.size()[3]  # (== w/16)

        out = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x)))  # (shape: (batch_size, num_classes, h/16, w/16))
        out = F.upsample(out, size=(x_h*2, x_w*2), mode="bilinear") # (shape: (batch_size, num_classes, h/8, w/8))
        
        out = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out))) # (shape: (batch_size, num_classes, h/8, w/8))
        
        out = F.relu(self.bn_conv_5x5_1(self.conv_1x1_2(out))) # (shape: (batch_size, num_classes, h/8, w/8))
        out = F.upsample(out, size=(x_h*4, x_w*4), mode="bilinear") # (shape: (batch_size, num_classes, h/4, w/4))
        
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, num_classes, h/4, w/4))
        
        out = F.relu(self.bn_conv_5x5_2(self.conv_5x5_2(out))) # (shape: (batch_size, num_classes, h/4, w/4))
        
        return out

class inst_decoder(nn.Module):
    def __init__(self, num_classes):
        super(inst_decoder, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(num_classes, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_1x1_2 = nn.Conv2d(256, 32, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        self.conv_1x1_3 = nn.Conv2d(128, 16, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)
        
        self.conv_5x5_1 = nn.Conv2d(32, 128, kernel_size=1)
        self.bn_conv_5x5_1 = nn.BatchNorm2d(256)

        self.conv_5x5_2 = nn.Conv2d(16, 128, kernel_size=1)
        self.bn_conv_5x5_2 = nn.BatchNorm2d(256)
        
    def forward(self, x):
        # (x has shape (batch_size, num_classes, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16.)
        # x is the output of aspp(feature_map)

        x_h = x.size()[2]  # (== h/16)
        x_w = x.size()[3]  # (== w/16)

        out = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x)))  # (shape: (batch_size, num_classes, h/16, w/16))
        out = F.upsample(out, size=(x_h*2, x_w*2), mode="bilinear") # (shape: (batch_size, num_classes, h/8, w/8))
        
        out = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out))) # (shape: (batch_size, num_classes, h/8, w/8))
        
        out = F.relu(self.bn_conv_5x5_1(self.conv_1x1_2(out))) # (shape: (batch_size, num_classes, h/8, w/8))
        out = F.upsample(out, size=(x_h*4, x_w*4), mode="bilinear") # (shape: (batch_size, num_classes, h/4, w/4))
        
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, num_classes, h/4, w/4))
        
        out  = F.relu(self.bn_conv_5x5_2(self.conv_5x5_2(out))) # (shape: (batch_size, num_classes, h/4, w/4))
        
        return out
