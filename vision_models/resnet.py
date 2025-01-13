"""
ResNet-50
"""
import torch
from torch import nn
import torch.nn.functional as F

class BottleNeckLayer(nn.Module):
    def __init__(self,input_channels:int,intermediate_channels:int,output_channels:int,block_number:int,stride:int):
        super(BottleNeckLayer,self).__init__()
        if block_number==0:
            self.convBlock1 = convBlock(input_channels,intermediate_channels,1,stride=stride)
            self.projection = convBlock(input_channels,output_channels,kernel_size=1,stride=stride)
        else:
            # Input to this block is the output of an identical previous block
            self.convBlock1 = convBlock(output_channels,intermediate_channels,1)
            self.projection = None # Identity projection
        self.convBlock2 = convBlock(intermediate_channels,intermediate_channels,3,padding=1)
        self.convBlock3 = convBlock(intermediate_channels,output_channels,1)

    def forward(self,input:torch.Tensor):
        x = self.convBlock1(input)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        if self.projection:
            return self.projection(input) + x
        else:
            return input + x

class BottleNeckBlock(nn.Module):
    def __init__(self,
                 input_channels:int,
                 intermediate_channels:int,
                 output_channels:int,
                 n_blocks:int,
                 stride:int = 2,
    ):
        super(BottleNeckBlock,self).__init__()
        self.block = nn.ModuleList(BottleNeckLayer(input_channels,intermediate_channels,output_channels,i,stride) for i in range(n_blocks))

    def forward(self,x):
        for layer in self.block:
            x = layer(x)
        return x 

class ResNet50(nn.Module):
    def __init__(self,config):
        super(ResNet50,self).__init__()
        self.conv1 = nn.Conv2d(config.conv1.input_channels,
                            config.conv1.output_channels,
                            kernel_size=config.conv1.kernel_size,
                            stride=config.conv1.stride,
                            padding=config.conv1.padding,
                            bias=False)
        self.maxpool1 = nn.MaxPool2d(config.maxpool1.kernel_size,
                                     stride=config.maxpool1.stride,
                                     padding=config.maxpool1.padding)
        
        self.block2 = BottleNeckBlock(config.block2.input_channels,
                                      config.block2.intermediate_channels,
                                      config.block2.output_channels,
                                      n_blocks=config.block2.n_blocks,
                                      stride=1)
        self.block3 = BottleNeckBlock(config.block3.input_channels,
                                      config.block3.intermediate_channels,
                                      config.block3.output_channels,
                                      n_blocks=config.block3.n_blocks) 
        self.block4 = BottleNeckBlock(config.block4.input_channels,
                                      config.block4.intermediate_channels,
                                      config.block4.output_channels,
                                      n_blocks=config.block4.n_blocks)
        self.block5 = BottleNeckBlock(config.block5.input_channels,
                                      config.block5.intermediate_channels,
                                      config.block5.output_channels,
                                      n_blocks=config.block5.n_blocks)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Conv2d(config.block5.output_channels, config.n_classes,kernel_size=1)

    def forward(self,input):
        x = self.conv1(input)
        x = self.maxpool1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x.squeeze()

class convBlock(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,kernel_size:int,padding:int=0,stride:int=1):
        super(convBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self,x):
        return F.relu(self.bn(self.conv(x)))

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
    def __init__(self,config):
        super(ResNetCifar,self).__init__()
        self.config = config
        self.layer_1 = convBlock(3,16,3)
        self.layer_2 = conv2nBlock(config.n,16,16,3,stride=1)
        self.layer_3 = conv2nBlock(config.n,16,32,3,stride=2)
        self.layer_4 = conv2nBlock(config.n,32,64,3,stride=2)
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

