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
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,bias=False)
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
        self.conv1 = nn.Conv2d(config.conv1.in_channels,
                            config.conv1.out_channels,
                            kernel_size=config.conv1.kernel_size,
                            stride=config.conv1.stride,
                            padding=config.conv1.padding,
                            bias=False)
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

class conv2Block(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,projection=False):
        super(conv2Block,self).__init__()
        self.projection = projection
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size,padding="same",bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if self.projection:
            self.projection_layer = nn.Conv2d(in_channels,out_channels,1,stride=2,bias=False)
            self.bn_p = nn.BatchNorm2d(out_channels)


    def forward(self,x):
        conv1 = F.relu(self.bn1(self.conv1(x)))
        if self.projection:
            x = F.relu(self.bn_p(self.projection_layer(x)))
        out = F.relu(x + self.bn2(self.conv2(conv1)))
        return out

class conv2nBlock(nn.Module):
    def __init__(self,n,in_channels,out_channels,kernel_size,stride):
        super(conv2nBlock,self).__init__()
        self.block_1 = conv2Block(in_channels,out_channels,kernel_size,stride=stride,projection=stride==2)
        self.blocks = nn.ModuleList(conv2Block(out_channels,out_channels,kernel_size) for _ in range(n-1))
    
    def forward(self,x):
        x = self.block_1(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResNetCifar(nn.Module):
    def __init__(self,n):
        super(ResNetCifar,self).__init__()
        self.layer_1 = convBlock(3,16,3)
        self.layer_2 = conv2nBlock(n,16,16,3,stride=1)
        self.layer_3 = conv2nBlock(n,16,32,3,stride=2)
        self.layer_4 = conv2nBlock(n,32,64,3,stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Conv2d(64,10,1,bias=False)
        print(f"No of parameters in the model: {self.get_num_params()}")

    def forward(self,x:torch.Tensor):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x.squeeze()

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

