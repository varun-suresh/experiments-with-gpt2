"""
ResNet-50
"""
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F

class convBlock(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,kernel_size:int,stride:int=1):
        super(convBlock, self).__init__()
        self.conv = nn.conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self,x):
        return F.relu(self.bn(self.conv(x)))


class BottleNeckLayer(nn.Module):
    def __init__(self,channels:int,is_conv2x:bool=False):
        if is_conv2x:
            self.conv1 = convBlock(channels,channels,1)
        else:
            self.conv1 = convBlock(2*channels,channels,1,stride=2)
        self.conv2 = convBlock(channels,channels,3)
        self.conv3 = convBlock(channels,4*channels,1)
    
    def forward(self,input:torch.Tensor):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        return input + x
        

class ResNet50(nn.Module):
    def __init__(self,config):
        super(ResNet50,self).__init__()
        self.conv1 = nn.conv2d(config.conv1.in_channels,
                            config.conv1.out_channels,
                            kernel_size=config.conv1.kernel_size,
                            stride=config.conv1.stride,
                            padding=config.conv1.padding)
        self.maxpool1 = nn.MaxPool2d(config.maxpool1.size,
                                     stride=config.maxpool1.stride)
        
        self.conv2 = nn.ModuleList(BottleNeckLayer(config.conv2.in_channels,is_conv2x=True) for _ in range(config.conv2.n_blocks))
        
        self.conv3 = nn.ModuleList(BottleNeckLayer(config.conv3.in_channels) for _ in range(config.conv3.n_blocks))
        self.conv4 = nn.ModuleList(BottleNeckLayer(config.conv4.in_channels) for _ in range(config.conv4.n_blocks))
        self.conv5 = nn.ModuleList(BottleNeckLayer(config.conv5.in_channels) for _ in range(config.conv5.n_blocks))
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(config.conv5.in_channels*4, config.n_classes)

    def forward(self,input):
        x = self.conv1(input)
        x = self.maxpool1(x)
        x =  self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x


