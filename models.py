import torch
import torch.nn as nn
import torchvision.models as models

from common import (CustomConv2d, ResidualBlock, OptimizedResidualBlock,global_pooling)

class RESNET101(nn.Module):
    def __init__(self,num_classes):
        super(RESNET101, self).__init__()
        self.model = models.resnet101(pretrained=True)
        #print(self.model)
        self.model.fc = nn.Linear(2048, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        y = self.model(x)
        y = self.softmax(y)
        return y

class Generator(nn.Module):
    def __init__(self):
        input_channels=3
        output_channels=3
        super(Generator, self).__init__()
        self.conv1 = conv_bn_relu(input_channels,64)
        self.conv2 = conv_bn_relu(64, 128)
        self.conv3 = conv_bn_relu(128, 256)
        self.conv4 = conv_bn_relu(256, 512)
        self.conv5 = conv_bn_relu(512, 1024)
        self.down_pooling = nn.MaxPool2d(2)
 
        self.up_pool6 = up_pooling(1024, 512)
        self.conv6 = conv_bn_relu(1024, 512)
        self.up_pool7 = up_pooling(512, 256)
        self.conv7 = conv_bn_relu(512, 256)
        self.up_pool8 = up_pooling(256, 128)
        self.conv8 = conv_bn_relu(256, 128)
        self.up_pool9 = up_pooling(128, 64)
        self.conv9 = conv_bn_relu(128, 64)
        self.conv10 = nn.Conv2d(64, output_channels, 1)
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
 
    def forward(self, x):
        x1 = self.conv1(x)
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)
        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)
        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)
        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)
 
        p6 = self.up_pool6(x5)
        x6 = torch.cat([p6, x4], dim=1)
        x6 = self.conv6(x6)
 
        p7 = self.up_pool7(x6)
        x7 = torch.cat([p7, x3], dim=1)
        x7 = self.conv7(x7)
 
        p8 = self.up_pool8(x7)
        x8 = torch.cat([p8, x2], dim=1)
        x8 = self.conv8(x8)
 
        p9 = self.up_pool9(x8)
        x9 = torch.cat([p9, x1], dim=1)
        x9 = self.conv9(x9)
 
        output = self.conv10(x9)
        output = torch.sigmoid(output)
 
        return output
 
def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
    )
 
def down_pooling():
    return nn.MaxPool2d(2)
 
def up_pooling(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class Discriminator(nn.Module):
    def __init__(self,
                 image_channels=3,
                 channels=128,
                 residual_factor=0.1,
                 pooling='mean'):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.image_channels = image_channels
        self.residual_factor = residual_factor
        self.pooling = pooling

        self.block1 = OptimizedResidualBlock(image_channels,
                                             channels,
                                             3,
                                             residual_factor=residual_factor)
        self.block2 = ResidualBlock(channels,
                                    channels,
                                    3,
                                    resample='down',
                                    residual_factor=residual_factor)
        self.block3 = ResidualBlock(channels,
                                    channels,
                                    3,
                                    resample=None,
                                    residual_factor=residual_factor)
        self.block4 = ResidualBlock(channels,
                                    channels,
                                    3,
                                    resample=None,
                                    residual_factor=residual_factor)
        self.relu5 = nn.ReLU()
        self.linear5 = nn.Linear(channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = input
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = self.relu5(output)
        output = global_pooling(output, self.pooling)
        out_dis = self.linear5(output)
        out_dis = self.sigmoid(out_dis)
        return out_dis.squeeze()

